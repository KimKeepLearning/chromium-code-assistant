#!/usr/bin/env bash
# Part 1 — Fetch the chromium mirror with milestone history (R136..R148).
# Reads repo.path / repo.url from config.yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO=$(python3 -c "import sys;sys.path.insert(0,'$ROOT/scripts');from common import load_config;print(load_config()['repo']['path'])")
URL=$(python3 -c "import sys;sys.path.insert(0,'$ROOT/scripts');from common import load_config;print(load_config()['repo']['url'])")

if [ ! -d "$REPO/.git" ]; then
  echo ">> Cloning $URL -> $REPO (this is large; may take a while)"
  git clone "$URL" "$REPO"
fi

cd "$REPO"
echo ">> Configuring milestone branch-heads fetch"
git config --add remote.origin.fetch '+refs/branch-heads/*:refs/remotes/branch-heads/*' || true
echo ">> Fetching tags & branch-heads"
git fetch origin --tags

echo ">> Milestone tags available (look for your from/to anchors):"
git tag | grep -E '^(136|148)\.' || echo "   (none matched — check that your mirror carries milestone tags/branch-heads)"
echo ">> Set versions.from / versions.to in config.yaml to actual tags above."
