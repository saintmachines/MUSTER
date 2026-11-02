#!/bin/bash
# Fetch the BEAM benchmark repository at the exact commit the paper's judge imports (unmodified evaluation code, MIT license)
# into harness/beam/BEAM, where harness/judge_beam.py looks for it (override the location with BEAM_REPO=<path>).
#   bash scripts/fetch_beam_repo.sh
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST="${BEAM_REPO:-$HERE/harness/beam/BEAM}"
COMMIT=b2da22e
if [ -d "$DEST/.git" ]; then
  echo "already present: $DEST ($(git -C "$DEST" rev-parse --short HEAD))"
else
  git clone --quiet https://github.com/mohammadtavakoli78/BEAM.git "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"
echo "BEAM repository at $(git -C "$DEST" rev-parse --short HEAD) in $DEST"
echo "judge entry point: $DEST/src/evaluation/compute_metrics.py"
