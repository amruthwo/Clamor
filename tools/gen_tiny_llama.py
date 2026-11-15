#!/usr/bin/env python3
"""Write a tiny random-weight Llama-architecture GGUF (f32) for engine tests.

Usage: gen_tiny_llama.py OUT.gguf [--layers 4] [--seed 0]
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
a = p.parse_args()

E, H, HKV, FF, V, CTX = 64, 4, 2, 128, 128, 64
HD = E // H
rng = np.random.default_rng(a.seed)

def w(*shape, scale=None):
    scale = scale or (1.0 / np.sqrt(shape[-1]))
    return (rng.standard_normal(shape) * scale).astype(np.float32)

wr = gguf.GGUFWriter(a.out, "llama")
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

wr.add_tensor("token_embd.weight", w(V, E, scale=0.5))
for i in range(a.layers):
    wr.add_tensor(f"blk.{i}.attn_norm.weight", (1 + 0.1 * rng.standard_normal(E)).astype(np.float32))
    wr.add_tensor(f"blk.{i}.attn_q.weight", w(H * HD, E))
    wr.add_tensor(f"blk.{i}.attn_k.weight", w(HKV * HD, E))
    wr.add_tensor(f"blk.{i}.attn_v.weight", w(HKV * HD, E))
    wr.add_tensor(f"blk.{i}.attn_output.weight", w(E, H * HD))
    wr.add_tensor(f"blk.{i}.ffn_norm.weight", (1 + 0.1 * rng.standard_normal(E)).astype(np.float32))
    wr.add_tensor(f"blk.{i}.ffn_gate.weight", w(FF, E))
    wr.add_tensor(f"blk.{i}.ffn_up.weight", w(FF, E))
    wr.add_tensor(f"blk.{i}.ffn_down.weight", w(E, FF))
wr.add_tensor("output_norm.weight", (1 + 0.1 * rng.standard_normal(E)).astype(np.float32))
wr.add_tensor("output.weight", w(V, E))

wr.write_header_to_file()
wr.write_kv_data_to_file()
wr.write_tensors_to_file()
wr.close()
print(f"wrote {a.out}")
