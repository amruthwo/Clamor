#!/usr/bin/env bash
# Package one release build and describe it for the release manifest.
# Usage: package-release.sh TARGET VARIANT CPU_LEVEL BIN_DIR OUT_DIR
# Writes OUT_DIR/clamor-VERSION-TARGET-VARIANT.{tar.gz,zip} and a matching .json fragment.
set -euo pipefail
target=$1 variant=$2 cpu=$3 bin_dir=$4
mkdir -p "$5" && out=$(cd "$5" && pwd)
version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
name="clamor-${version}-${target}-${variant}"
exe=""; [[ "$target" == *windows* ]] && exe=".exe"
stage="$(mktemp -d)/$name"
mkdir -p "$stage"
for b in clamor-node clamor-muster; do cp "$bin_dir/$b$exe" "$stage/"; done
cp README.md LICENSE "$stage/"
if [[ -n "$exe" ]]; then
  file="$name.zip"; (cd "$(dirname "$stage")" && 7z a -tzip -bso0 "$out/$file" "$name" >/dev/null)
else
  file="$name.tar.gz"; tar czf "$out/$file" -C "$(dirname "$stage")" "$name"
fi
sha=$( (sha256sum "$out/$file" 2>/dev/null || shasum -a 256 "$out/$file") | cut -d' ' -f1)
size=$(wc -c < "$out/$file" | tr -d ' ')
printf '{"target":"%s","variant":"%s","cpu":"%s","file":"%s","sha256":"%s","size":%s}\n' "$target" "$variant" "$cpu" "$file" "$sha" "$size" > "$out/$name.json"
echo "packaged $out/$file ($size bytes, sha256 $sha)"
