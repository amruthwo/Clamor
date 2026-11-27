#!/usr/bin/env bash
# Describe each app installer in DIR for the release manifest (a .json next to each file),
# like package-release.sh does for the command-line archives.
# Usage: describe-installers.sh DIR TARGET CPU_LEVEL
set -euo pipefail
dir=$1 target=$2 cpu=$3
for f in "$dir"/*; do
  case "$f" in
    *.deb) kind=deb ;;
    *.rpm) kind=rpm ;;
    *.AppImage) kind=appimage ;;
    *.flatpak) kind=flatpak ;;
    *.dmg) kind=dmg ;;
    *-setup.exe) kind=nsis ;;
    *.apk) kind=apk ;;
    *) continue ;;
  esac
  sha=$( (sha256sum "$f" 2>/dev/null || shasum -a 256 "$f") | cut -d' ' -f1)
  size=$(wc -c < "$f" | tr -d ' ')
  printf '{"target":"%s","variant":"app","cpu":"%s","file":"%s","sha256":"%s","size":%s,"kind":"%s"}\n' "$target" "$cpu" "$(basename "$f")" "$sha" "$size" "$kind" > "$f.json"
  echo "described $(basename "$f") ($kind)"
done
