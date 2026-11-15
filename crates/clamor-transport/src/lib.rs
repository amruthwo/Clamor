//! Clamor data-plane transport: how activations move between pipeline stages.
//!
//! * `frame`  - fixed 40-byte binary header + raw payload (no JSON on the hot path).
//! * `link`   - one TCP connection per stage pair: handshake, epoch fencing, credit-based
//!              flow control so a slow stage back-pressures its upstream instead of buffering.
//! * `stage`  - ring pipeline. Stage 0 (head) owns the embedding and the request table, the
//!              last stage (tail) samples a token and sends it straight back to the head, so
//!              a token costs one hop per stage rather than a round trip through a hub.
//!              Several requests can be in flight at once, which keeps every stage busy.
//! * `codec`  - f16 / int8 activation encoding to halve or quarter link traffic.
//! * `mock`   - deterministic stand-in engine used by tests and the benchmark.
//!
//! The transport knows nothing about the inference engine: anything implementing
//! [`stage::StageExecutor`] can sit in a stage.

pub mod codec;
pub mod frame;
pub mod link;
pub mod mock;
pub mod stage;
