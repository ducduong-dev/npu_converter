#!/usr/bin/env bash
# Extract the NXP eIQ Neutron SDK zip into vendor/ and print the env var to use.
#
# The SDK is proprietary (LA_OPT_NXP license) and must not be committed or baked
# into a published image. This script just unpacks a copy you already obtained.
#
# Usage:
#   scripts/setup_sdk.sh path/to/eiq-neutron-sdk-linux-3.1.3.zip [dest_dir]
set -euo pipefail

ZIP="${1:-}"
DEST="${2:-vendor/eiq-neutron-sdk}"

if [[ -z "$ZIP" || ! -f "$ZIP" ]]; then
    echo "usage: $0 <eiq-neutron-sdk-*.zip> [dest_dir]" >&2
    exit 1
fi

mkdir -p "$DEST"
unzip -o -q "$ZIP" -d "$DEST"
chmod +x "$DEST"/bin/* 2>/dev/null || true

ABS_DEST="$(cd "$DEST" && pwd)"
VERSION="$("$ABS_DEST/bin/neutron-converter" --version 2>/dev/null | head -1 || echo '?')"

echo "Extracted: $VERSION"
echo "SDK ready at: $ABS_DEST"
echo
echo "Point the converter service at it with:"
echo "  export NEUTRON_SDK_DIR=\"$ABS_DEST\""
