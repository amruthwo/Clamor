# Clamor: working notes for Claude Code

Rust workspace that pools idle LAN machines to run models too big for any one of them.
Read `README.md` first, then this file. Design doc lives in Matt's Claude Docs ("Clamor design").

## Crates
- `clamor-proto`: control-plane wire types (JSON frames over TCP 7420).
- `clamor-muster`: probe/benchmark/plan CLI + library. Planner picks stage splits and ring/hub topology.
- `clamor-node`: coordinator + member + portal (7421). `clamor-node coordinator|join`.
- `clamor-transport`: own binary ring transport between stages; engine boundary is `trait StageExecutor`.
- `clamor-engine-ggml`: vendored ggml (`vendor/ggml`, pinned in `vendor/ggml.VERSION`) built by CMake from build.rs.
  Backends are cargo features: `cuda vulkan metal hip opencl`. Bindings are checked in; regenerate with `scripts/gen-bindings.sh`.

## State (v0.3)
Working and tested: protocol, muster, planner, coordinator/portal, ring transport with a mock engine,
ggml builds and runs matmul on CPU, GGUF metadata read.
Step 1 done: `LlamaStage` (crates/clamor-engine-ggml/src/llama.rs) runs layers [a,b) of an f32 Llama GGUF with a
per-request KV cache; matches tools/ref_forward.py to ~1e-6 on CPU, per layer, for full, 2- and 3-stage splits,
and incremental decode (tests/llama_ref.rs). Tied lm_head (no output.weight) supported.
Step 2 done: `GgmlExecutor` (src/executor.rs) puts a `LlamaStage` behind `StageExecutor`: f16/int8 activations
on the wire, tail samples (greedy, or temperature/top-k with a per-request seeded RNG), EOS ids from GGUF.
tests/ring_e2e.rs runs real rings on loopback: 1-4 stages match numpy greedy generation, and the ring is
bit-exact with the same executors called in-process; cancel, EOS, concurrency, context overflow.
NOT done: node wiring (ShardAssign), tokenizer, quantized weights, real models.

## Next steps, in order
1. DONE. `tools/gen_tiny_llama.py` + `tools/ref_forward.py` give a random-weight Llama model and numpy reference
   (per-layer hidden states + logits, saved as .npz). Implement in `clamor-engine-ggml` a layer-range graph:
   embed (head only) → layers [a,b) with KV cache → output norm + lm_head (tail only). Match the reference
   per layer (f32 tolerance ~1e-4), then check split-equals-full (layers 0..2 then 2..4 == 0..4).
   Convention is ggml's (RoPE on adjacent pairs, weights [out,in]).
2. DONE. `GgmlExecutor: StageExecutor` (crates/clamor-transport/src/stage.rs defines the trait). One KV cache per request id; `release()` frees it.
3. Wire executor into `clamor-node` via the `ShardAssign` message; add a real tokenizer path (GGUF vocab).
4. Quantized weights (Q4_K etc.) loaded from real GGUF; then real small model (e.g. a ~1B Llama) end to end on 2 local processes.
5. Per-device GPU benchmark → `BenchReport` → planner (planner currently ignores accelerators until benchmarked).
6. Later: pairwise link measurement, mDNS discovery, mTLS, Android app wrapper.

## Engine notes
- `LlamaStage::forward(req, pos, input)`: `pos` must equal tokens already cached for `req` (no gaps/rewinds).
  Head takes `Tokens`, others take `Hidden` (`n_tokens * n_embd` f32, token-major); tail returns logits.
- KV cache is f32, `n_ctx` cells per request, written with `ggml_set_rows` so reads depend on the write.
- Reference tests skip without python3+numpy+gguf unless `CLAMOR_REQUIRE_PY=1` (CI sets it).
- Before computing, every graph node is checked with `ggml_backend_supports_op`; a device that can't run one
  returns `Error::Unsupported` and the tests skip it (CPU must always pass). Hosted macOS runners expose an
  "Apple Paravirtual device" without simdgroup reduction, so RMS_NORM/SOFT_MAX are unsupported there:
  the macos CI job does not prove Metal inference works. Verify on a real Mac.
- Device order: GPUs first. Index 0 is the CPU only in a CPU-only build.
- Engine errors fail one request, not the ring: a failing stage releases the request and sends a `Token` frame
  with `FLAG_FAILED` (payload = message) on around the ring; the head ends that request `Failed(msg)` and
  cancels it everywhere. Before this, any `forward` error ended the stage loop and broke every request.
- The tail computes logits for the last token only (`Logits::Last`); `Logits::All` exists for tests.
- `tools/ref_forward.py --generate N` gives numpy greedy tokens (and prints the smallest top-1/top-2 gap,
  which bounds how much activation rounding greedy text can tolerate).

## Rules
- Do not claim platform support that CI has not exercised. CI build-only jobs prove compilation only.
- Keep `cargo test --workspace --release` green; every bug found gets a regression test.
- Don't edit `vendor/ggml` except by re-vendoring (update `vendor/ggml.VERSION`). Never put a file named `version` (any case) inside `vendor/ggml`: on case-insensitive macOS it shadows the C++ `<version>` header. Don't change checked-in bindings by hand.
