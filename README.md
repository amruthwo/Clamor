# Clamor

Pool the idle computers on your network to run local AI no single machine could.
Design spec: see the Clamor design spec doc (sections 1-14).

## Try it (two terminals, or two machines)

    cargo build --release

    # machine A: start the swarm (control :7420, web portal :7421)
    ./target/release/clamor-node coordinator --swarm home --token SECRET

    # machine B: join (benchmarks itself, link is measured by the coordinator)
    ./target/release/clamor-node join <machine-A-ip> --token SECRET

    # open http://<machine-A-ip>:7421/

## Muster by hand (no swarm needed)

    ./target/release/clamor-muster probe --out a.json                 # on each machine
    ./target/release/clamor-muster serve                              # on the coordinator
    ./target/release/clamor-muster probe --coordinator <ip>:7422 --out b.json
    ./target/release/clamor-muster plan a.json b.json --quant q4 --ctx 8192

Always use --release: debug builds benchmark far too slow to plan with.

## Transport (data plane)

`clamor-transport` moves activations between pipeline stages over a ring, with its own binary
framing, credit flow control, epoch fencing and cancel. It is engine-agnostic (`StageExecutor`).

    cargo test --release -p clamor-transport
    cargo run --release -p clamor-transport --bin transport-bench -- --stages 3 --concurrency 1,2,4,8

## Status
v0.1: protocol, muster, coordinator, portal, ring transport (mock engine only).
Not yet: a real inference engine behind `StageExecutor`, task execution, TLS, Android.

## ggml engine (v0.3 scaffold)

`crates/clamor-engine-ggml` vendors ggml and exposes device enumeration, a matmul check/benchmark and GGUF metadata.

```bash
cargo run -p clamor-engine-ggml --release --example devices                  # CPU
cargo run -p clamor-engine-ggml --release --features vulkan --example devices
```

Not yet implemented: the Llama layer-range graph and the `StageExecutor` that runs it (see `CLAUDE.md`).
Building on an Asahi laptop and pushing to GitHub: see `ASAHI_SETUP.md`.
