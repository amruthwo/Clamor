#!/usr/bin/env bash
# Build clamor-node and put it where the desktop app's bundler expects it
# (binaries/clamor-node-<target triple>). Extra arguments go to cargo, e.g. --features vulkan.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
target=${CLAMOR_TARGET:-$(rustc -vV | sed -n 's/^host: //p')}
(cd "$root" && cargo build --release -p clamor-node --target "$target" "$@")
ext=""; case "$target" in *windows*) ext=".exe";; esac
mkdir -p "$here/binaries"
cp "$root/target/$target/release/clamor-node$ext" "$here/binaries/clamor-node-$target$ext"
echo "binaries/clamor-node-$target$ext ready"
