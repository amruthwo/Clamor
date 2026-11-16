#!/usr/bin/env bash
# Build the Flatpak from the .deb made by `cargo tauri build` (run from apps/desktop first).
# Optional argument: the folder holding the .deb (default: the native build's bundle/deb).
# Output: Clamor-<version>-<arch>.flatpak in this folder.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
debdir=${1:-$here/../target/release/bundle/deb}
deb=$(ls -t "$debdir"/*.deb | head -1)
cp "$deb" "$here/clamor.deb"
# Flathub's shared module recipes (tray icon library), pinned.
modules_commit=c91a81101cf9bcf9fc8132e8c3ebed741ad8bae6
if [ "$(git -C "$here/shared-modules" rev-parse HEAD 2>/dev/null)" != "$modules_commit" ]; then
  rm -rf "$here/shared-modules"
  git clone -q https://github.com/flathub/shared-modules.git "$here/shared-modules"
  git -C "$here/shared-modules" checkout -q "$modules_commit"
fi
flatpak --user remote-add --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo
flatpak-builder --user --install-deps-from=flathub --force-clean --repo="$here/repo" "$here/build" "$here/io.github.amruthwo.Clamor.yml"
arch=$(flatpak --default-arch)
version=$(sed -n 's/.*"version": "\(.*\)".*/\1/p' "$here/../tauri.conf.json" | head -1)
flatpak build-bundle "$here/repo" "$here/Clamor-$version-$arch.flatpak" io.github.amruthwo.Clamor
echo "$here/Clamor-$version-$arch.flatpak"
