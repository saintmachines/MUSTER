#!/bin/bash
# Fetch the MemoryAgentBench repository (MIT license) at the commit the paper's harness read its query templates, judge
# prompt and scorers from, into harness/mab_repo (override the location with MAB_REPO=<path>), and check the four files
# the harness reads against their sha256.
#   bash scripts/fetch_mab_repo.sh
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST="${MAB_REPO:-$HERE/harness/mab_repo}"
COMMIT=455306dcabc3842526eb83cd4e225e5d486c5c5d
if [ -d "$DEST/.git" ]; then
  echo "already present: $DEST ($(git -C "$DEST" rev-parse --short HEAD))"
else
  git clone --quiet https://github.com/HUST-AI-HYZ/MemoryAgentBench.git "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"
cd "$DEST"
shasum -a 256 -c <<'SUMS'
148c40d48d19f155ae845482c4417ba59cfa7ae4e194019509e023bd3a8755dd  utils/templates.py
d77976be409298970614d477a9d8003850caddb0510e56a7e821a037d98493a2  utils/eval_other_utils.py
ceca59bddea4203a78b417be635e63fe3f685bd957e3df12150978e21972c540  llm_based_eval/longmem_qa_evaluate.py
6951d19644cdb1972bcf6cdb272f4757aa995ae07099bf74012925158c376dd4  llm_based_eval/summarization_evaluate.py
SUMS
echo "MemoryAgentBench at $(git rev-parse --short HEAD) in $DEST"
