#!/usr/bin/env bash
# Part 1 — Verify/update the (existing) vibe chromium checkout and its milestone
# branches. Reads repo.path and versions.* from config.yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cfg() { python3 -c "import sys;sys.path.insert(0,'$ROOT/scripts');from common import load_config;print(load_config()$1)"; }
REPO=$(cfg "['repo']['path']")
FROM=$(cfg "['versions']['from_ref']")
TO=$(cfg "['versions']['to_ref']")

if [ ! -d "$REPO/.git" ]; then
  echo "!! No git repo at $REPO — set repo.path in config.yaml"; exit 1
fi

echo ">> Fetching branches & tags (updates ${FROM} / ${TO})"
git -C "$REPO" fetch origin --tags --prune

echo ">> Milestone branches available:"
git -C "$REPO" branch -r | grep -E 'origin/R[0-9]+$' || echo "   (none matched origin/R<NN>)"

echo ">> Verifying configured anchors resolve:"
for ref in "$FROM" "$TO"; do
  if git -C "$REPO" rev-parse --verify -q "$ref" >/dev/null; then
    echo "   OK  $ref -> $(git -C "$REPO" rev-parse --short "$ref")"
  else
    echo "   !!  $ref does NOT resolve — fix versions.* in config.yaml"; exit 1
  fi
done

echo ">> Commit counts:"
echo "   ${FROM}..${TO} (new in ${TO}): $(git -C "$REPO" rev-list --count "${FROM}..${TO}")"
echo "   merge-base: $(git -C "$REPO" merge-base "$FROM" "$TO" | cut -c1-12)"
echo ">> Ready. Next: python scripts/02_extract_commits.py"
