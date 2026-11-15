#!/usr/bin/env python3
"""Numpy reference forward pass for a gen_tiny_llama.py model.

Saves the hidden state after every layer and the final logits so the Rust engine can
check (a) full-model equality and (b) that running layers [0,k) then [k,n) on two
"stages" gives the same result as one pass.

Usage: ref_forward.py MODEL.gguf OUT.npz --tokens 1,5,9,23
"""
import argparse
import numpy as np
import gguf

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("out")
p.add_argument("--tokens", default="1,5,9,23,42,7")
a = p.parse_args()
toks = [int(t) for t in a.tokens.split(",")]

r = gguf.GGUFReader(a.model)
T = {t.name: np.array(t.data, dtype=np.float32).reshape(tuple(reversed(t.shape.tolist()))) for t in r.tensors}
def kv(k):
    f = r.fields[k]
    return f.contents()
n_layer = int(kv("llama.block_count")); E = int(kv("llama.embedding_length"))
H = int(kv("llama.attention.head_count")); HKV = int(kv("llama.attention.head_count_kv"))
eps = float(kv("llama.attention.layer_norm_rms_epsilon")); base = float(kv("llama.rope.freq_base"))
HD = E // H

def rms(x, g): return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g
def rope(x, pos):  # x: [T, heads, HD], adjacent pairs (ggml NORM)
    i = np.arange(HD // 2)
    ang = pos[:, None] * base ** (-2.0 * i / HD)
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out
def silu(x): return x / (1 + np.exp(-x))

n = len(toks); pos = np.arange(n)
x = T["token_embd.weight"][toks]
hidden = []
mask = np.triu(np.full((n, n), -np.inf, dtype=np.float32), 1)
for l in range(n_layer):
    g = lambda s: T[f"blk.{l}.{s}.weight"]
    h = rms(x, g("attn_norm"))
    q = rope((h @ g("attn_q").T).reshape(n, H, HD), pos)
    k = rope((h @ g("attn_k").T).reshape(n, HKV, HD), pos)
    v = (h @ g("attn_v").T).reshape(n, HKV, HD)
    rep = H // HKV
    k = np.repeat(k, rep, axis=1); v = np.repeat(v, rep, axis=1)
    sc = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(HD) + mask
    sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
    att = np.einsum("hqk,khd->qhd", sc, v).reshape(n, H * HD)
    x = x + att @ g("attn_output").T
    h = rms(x, g("ffn_norm"))
    x = x + (silu(h @ g("ffn_gate").T) * (h @ g("ffn_up").T)) @ g("ffn_down").T
    hidden.append(x.copy())
logits = rms(x, T["output_norm.weight"]) @ T["output.weight"].T
np.savez(a.out, tokens=np.array(toks), hidden=np.stack(hidden), logits=logits)
print("next-token argmax:", int(logits[-1].argmax()), "saved", a.out)
