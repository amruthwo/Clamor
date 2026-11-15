#!/usr/bin/env bash
# Regenerate crates/clamor-engine-ggml/src/bindings.rs from vendored ggml headers.
# Requires: cargo install bindgen-cli, libclang.
set -euo pipefail
cd "$(dirname "$0")/.."
bindgen crates/clamor-engine-ggml/wrapper/wrapper.h \
  --use-core --no-layout-tests --no-doc-comments \
  --allowlist-function '(ggml|gguf)_.*' \
  --allowlist-type '(ggml|gguf)_.*' \
  --allowlist-var '(GGML|GGUF)_.*' \
  --rustified-enum 'ggml_(type|op|backend_dev_type|status|prec)' \
  --rustified-enum 'gguf_type' \
  --no-copy 'ggml_.*' \
  -o crates/clamor-engine-ggml/src/bindings.rs \
  -- -Ivendor/ggml/include
echo "wrote crates/clamor-engine-ggml/src/bindings.rs"
