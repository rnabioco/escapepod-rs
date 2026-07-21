#!/bin/bash
# Regenerate all VHS demo gifs under docs/images/ from the tapes in docs/tapes/.
#
# macOS laptop ONLY. VHS renders by driving a headless Chromium (via ttyd) and
# ffmpeg — far too heavy for the 2-core cluster login node, and it must never run
# on a compute node. Provisioning is via Homebrew, not pixi/conda, so there is no
# way to launch this on the cluster by accident.
#
# One-time setup:
#   brew install vhs                    # pulls in ttyd + ffmpeg
#   brew install --cask font-fira-code  # the tapes pin `Set FontFamily "Fira Code"`
#
# Run from anywhere: ./docs/tapes/generate.sh
# The tapes read sample POD5 under data/drna/ (gitignored) — that data must be
# present locally for the gifs to render.

set -euo pipefail

if [[ "$(uname)" != "Darwin" ]]; then
    echo "error: tape regeneration runs on a local macOS laptop only, not the cluster." >&2
    echo "       VHS drives a headless Chromium + ffmpeg and will overwhelm the login node." >&2
    exit 1
fi

if ! command -v vhs >/dev/null 2>&1; then
    echo "error: vhs not found. Install it with:  brew install vhs" >&2
    echo "       (this also installs its ttyd + ffmpeg dependencies)" >&2
    exit 1
fi

# Fira Code is required by every tape. Warn if it's missing rather than hard-fail —
# vhs will surface its own font error, but this points at the fix first.
if ! ls "$HOME"/Library/Fonts/FiraCode* >/dev/null 2>&1 \
   && ! /usr/bin/find /Library/Fonts -maxdepth 1 -iname 'FiraCode*' 2>/dev/null | grep -q .; then
    echo "warning: Fira Code not found in ~/Library/Fonts or /Library/Fonts." >&2
    echo "         Install it with:  brew install --cask font-fira-code" >&2
fi

cd "$(dirname "$0")/../.."

# Build the release binary and put it on PATH so the gifs show current CLI output.
cargo build --release
export PATH="$PWD/target/release:$PATH"

for tape in docs/tapes/*.tape; do
    echo "Generating: $tape"
    vhs "$tape"
done

echo "Done! Gifs are in docs/images/"
