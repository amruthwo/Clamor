#!/usr/bin/env python3
"""Numpy reference forward pass for a gen_tiny_llama.py model.

Saves the hidden state after every layer and the final logits so the Rust engine can
check (a) full-model equality and (b) that running layers [0,k) then [k,n) on two
"stages" gives the same result as one pass.

Usage: ref_forward.py MODEL.gguf OUT.npz --tokens 1,5,9,23 [--raw DIR]

--generate N appends N greedy tokens (argmax, full recompute each step).
--raw also writes DIR/generated.txt (comma-separated), DIR/hidden.f32 ([layers, tokens, n_embd]) and DIR/logits.f32
([tokens, n_vocab]) as little-endian f32, for tests that cannot read .npz.
"""
import argparse
import numpy as np
import gguf

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("out")
p.add_argument("--tokens", default="1,5,9,23,42,7")
p.add_argument("--raw", help="also write raw little-endian f32 files to this directory")
p.add_argument("--generate", type=int, default=0, help="also generate this many tokens greedily (written to generated.txt with --raw)")
a = p.parse_args()
toks = [int(t) for t in a.tokens.split(",")]

r = gguf.GGUFReader(a.model)
def tensor_f32(t):
    """Any GGUF tensor as f32 numpy, shaped [out, in] (quantized types dequantized by the gguf package)."""
    if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        x = np.array(t.data, dtype=np.float32)
    else:
        x = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
    return x.reshape(tuple(reversed(t.shape.tolist())))
T = {t.name: tensor_f32(t) for t in r.tensors}
def kv(k):
    f = r.fields[k]
    return f.contents()
arch = str(kv("general.architecture"))
assert arch in ("llama", "qwen2"), f"unsupported architecture {arch}"
P = arch + "."
n_layer = int(kv(P + "block_count")); E = int(kv(P + "embedding_length"))
H = int(kv(P + "attention.head_count")); HKV = int(kv(P + "attention.head_count_kv"))
eps = float(kv(P + "attention.layer_norm_rms_epsilon")); base = float(kv(P + "rope.freq_base"))
HD = E // H
# Qwen2 rotates the two halves of each head (NEOX); llama.cpp's converter permutes Llama's q/k
# weights so Llama rotates adjacent pairs (NORM).
NEOX = arch == "qwen2"
# Llama 3 RoPE scaling: frequency i is divided by rope_freqs[i].
FF = np.array(next(t.data for t in r.tensors if t.name == "rope_freqs.weight"), dtype=np.float64) if any(t.name == "rope_freqs.weight" for t in r.tensors) else np.ones(HD // 2)

def rms(x, g): return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g
def rope(x, pos):  # x: [T, heads, HD]
    i = np.arange(HD // 2)
    ang = pos[:, None] * base ** (-2.0 * i / HD) / FF
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    out = np.empty_like(x)
    if NEOX:  # pairs (i, i + HD/2)
        x0, x1 = x[..., : HD // 2], x[..., HD // 2 :]
        out[..., : HD // 2] = x0 * c - x1 * s
        out[..., HD // 2 :] = x0 * s + x1 * c
    else:  # adjacent pairs (2i, 2i+1)
        x0, x1 = x[..., 0::2], x[..., 1::2]
        out[..., 0::2] = x0 * c - x1 * s
        out[..., 1::2] = x0 * s + x1 * c
    return out
def bias(l, name):
    return T.get(f"blk.{l}.{name}.bias", 0.0)
def silu(x): return x / (1 + np.exp(-x))

def forward(toks):
    """Full forward pass (no cache). Returns per-layer hidden states and logits for every token."""
    n = len(toks); pos = np.arange(n)
    x = T["token_embd.weight"][toks]
    hidden = []
    mask = np.triu(np.full((n, n), -np.inf, dtype=np.float32), 1)
    for l in range(n_layer):
        g = lambda s: T[f"blk.{l}.{s}.weight"]
        h = rms(x, g("attn_norm"))
        q = rope((h @ g("attn_q").T + bias(l, "attn_q")).reshape(n, H, HD), pos)
        k = rope((h @ g("attn_k").T + bias(l, "attn_k")).reshape(n, HKV, HD), pos)
        v = (h @ g("attn_v").T + bias(l, "attn_v")).reshape(n, HKV, HD)
        rep = H // HKV
        k = np.repeat(k, rep, axis=1); v = np.repeat(v, rep, axis=1)
        sc = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(HD) + mask
        sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
        att = np.einsum("hqk,khd->qhd", sc, v).reshape(n, H * HD)
        x = x + att @ g("attn_output").T
        h = rms(x, g("ffn_norm"))
        x = x + (silu(h @ g("ffn_gate").T) * (h @ g("ffn_up").T)) @ g("ffn_down").T
        hidden.append(x.copy())
    lm_head = T.get("output.weight", T["token_embd.weight"])  # tied when output.weight is absent
    return hidden, rms(x, T["output_norm.weight"]) @ lm_head.T

hidden, logits = forward(toks)

# Greedy generation: append the argmax and rerun, recomputing everything (no KV cache).
generated, seq, margins = [], list(toks), []
for _ in range(a.generate):
    _, lg = forward(seq)
    top2 = np.sort(lg[-1])[-2:]
    margins.append(float(top2[1] - top2[0]))
    nxt = int(lg[-1].argmax())
    generated.append(nxt); seq.append(nxt)
np.savez(a.out, tokens=np.array(toks), hidden=np.stack(hidden), logits=logits, generated=np.array(generated, dtype=np.int64))
if a.raw:
    import os
    os.makedirs(a.raw, exist_ok=True)
    np.stack(hidden).astype("<f4").tofile(os.path.join(a.raw, "hidden.f32"))
    logits.astype("<f4").tofile(os.path.join(a.raw, "logits.f32"))
    with open(os.path.join(a.raw, "generated.txt"), "w", encoding="utf-8") as f:
        f.write(",".join(map(str, generated)))
print("next-token argmax:", int(logits[-1].argmax()), "saved", a.out)
if generated:
    print("greedy:", generated, "smallest top-1/top-2 logit gap: %.4f" % min(margins))
