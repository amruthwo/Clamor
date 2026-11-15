#!/usr/bin/env python3
"""Write a tiny random-weight Llama-architecture GGUF (f32) for engine tests.

Usage: gen_tiny_llama.py OUT.gguf [--layers 4] [--seed 0] [--tied] [--bytes-vocab] [--arch qwen2] [--rope-freqs]
--tied omits output.weight, so lm_head reuses token_embd.weight (as many small models do).
Weights are in ggml convention (RoPE on adjacent pairs), so ref_forward.py and the
Rust graph must agree on exactly this definition.
"""
import argparse
import numpy as np
import gguf

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--layers", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tied", action="store_true", help="omit output.weight (lm_head tied to token_embd)")
p.add_argument("--arch", choices=["llama", "qwen2"], default="llama",
               help="qwen2: q/k/v biases and NEOX RoPE (as Qwen2/2.5)")
p.add_argument("--rope-freqs", action="store_true", help="add a rope_freqs.weight tensor (Llama 3 RoPE scaling)")
p.add_argument("--bytes-vocab", action="store_true",
               help="add a byte-level BPE tokenizer: the 256 byte tokens + ChatML specials, no merges (vocab 259)")
a = p.parse_args()

E, H, HKV, FF, V, CTX = 64, 4, 2, 128, 128, 64
if a.bytes_vocab:
    V = 259
HD = E // H
rng = np.random.default_rng(a.seed)

def w(*shape, scale=None):
    scale = scale or (1.0 / np.sqrt(shape[-1]))
    return (rng.standard_normal(shape) * scale).astype(np.float32)

wr = gguf.GGUFWriter(a.out, a.arch)
wr.add_name("clamor-tiny-llama")
wr.add_context_length(CTX)
wr.add_embedding_length(E)
wr.add_block_count(a.layers)
wr.add_feed_forward_length(FF)
wr.add_head_count(H)
wr.add_head_count_kv(HKV)
wr.add_layer_norm_rms_eps(1e-5)
wr.add_rope_dimension_count(HD)
wr.add_rope_freq_base(10000.0)
wr.add_vocab_size(V)
if a.bytes_vocab:
    # GPT-2's byte alphabet, in byte order, then the ChatML control tokens.
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    enc, n = {}, 0
    for b in range(256):
        if b in printable:
            enc[b] = chr(b)
        else:
            n += 1
            enc[b] = chr(255 + n)
    tokens = [enc[b] for b in range(256)] + ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    wr.add_tokenizer_model("gpt2")
    wr.add_tokenizer_pre("gpt2")
    wr.add_token_list(tokens)
    wr.add_token_types([1] * 256 + [3, 3, 3])
    wr.add_token_merges([])
    wr.add_bos_token_id(256)
    wr.add_eos_token_id(258)

wr.add_tensor("token_embd.weight", w(V, E, scale=0.5))
for i in range(a.layers):
    wr.add_tensor(f"blk.{i}.attn_norm.weight", (1 + 0.1 * rng.standard_normal(E)).astype(np.float32))
    wr.add_tensor(f"blk.{i}.attn_q.weight", w(H * HD, E))
    wr.add_tensor(f"blk.{i}.attn_k.weight", w(HKV * HD, E))
    wr.add_tensor(f"blk.{i}.attn_v.weight", w(HKV * HD, E))
    if a.arch == "qwen2":
        for nm, rows in (("attn_q", H * HD), ("attn_k", HKV * HD), ("attn_v", HKV * HD)):
            wr.add_tensor(f"blk.{i}.{nm}.bias", (0.5 * rng.standard_normal(rows)).astype(np.float32))
    wr.add_tensor(f"blk.{i}.attn_output.weight", w(E, H * HD))
    wr.add_tensor(f"blk.{i}.ffn_norm.weight", (1 + 0.1 * rng.standard_normal(E)).astype(np.float32))
    wr.add_tensor(f"blk.{i}.ffn_gate.weight", w(FF, E))
    wr.add_tensor(f"blk.{i}.ffn_up.weight", w(FF, E))
    wr.add_tensor(f"blk.{i}.ffn_down.weight", w(E, FF))
if a.rope_freqs:
    # Llama 3 style: low frequencies slowed down, high ones untouched.
    wr.add_tensor("rope_freqs.weight", np.linspace(1.0, 8.0, HD // 2).astype(np.float32))
wr.add_tensor("output_norm.weight", (1 + 0.1 * rng.standard_normal(E)).astype(np.float32))
if not a.tied:
    wr.add_tensor("output.weight", w(V, E))

wr.write_header_to_file()
wr.write_kv_data_to_file()
wr.write_tensors_to_file()
wr.close()
print(f"wrote {a.out}")
