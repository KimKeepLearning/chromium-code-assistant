"""Part 2B — Extract commits for versions.from_ref..to_ref, classified by origin.

The raw range is huge (170k+), mostly upstream Chromium. Strategy:
  1. cheap metadata pass via `git log` over include_dirs only (pathspec-filtered);
  2. classify each commit as 'vibe' (downstream) or 'official' (upstream Chromium);
  3. drop bot/DEPS-roll noise;
  4. keep ALL vibe commits + a chronologically-spread SAMPLE of official ones;
  5. fetch diffs only for the selected commits.

Output: data/commits.jsonl  — feeds the SFT dataset (Part 3) and RAG 'evolution' (Part 5).
"""
import json
import subprocess
from common import load_config, data_path

cfg = load_config()
REPO = cfg["repo"]["path"]
v, prov = cfg["versions"], cfg["provenance"]
rng = f"{v['from_ref']}..{v['to_ref']}"
include_dirs = cfg["include_dirs"]
exts = tuple(cfg["extensions"])
ext_pathspecs = [f"*{e}" for e in exts]
max_diff = cfg["dataset"]["max_diff_chars"]

US, RS = "\x1f", "\x1e"  # field / record separators


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args],
                          capture_output=True, text=True, errors="ignore").stdout


def classify(email, body):
    b = body.lower()
    if any(m.lower() in b for m in prov["official_body_markers"]):
        return "official"
    if any(email.lower().endswith(d.lower()) for d in prov["official_email_domains"]):
        return "official"
    return "vibe"


def is_noise(name, email, subject):
    e = email.lower()
    if any(p.lower() in e for p in prov["skip_email_patterns"]):
        return True
    return any(p.lower() in subject.lower() for p in prov["skip_subject_patterns"])


fmt = US.join(["%H", "%an", "%ae", "%aI", "%s", "%b"]) + RS


def metadata_pass(pathspec_dirs=None):
    """Cheap (no-diff) git log pass; returns parsed, noise-filtered commits."""
    args = ["log", rng, "--no-merges", f"--format={fmt}"]
    if pathspec_dirs:
        args += ["--", *pathspec_dirs]
    items = []
    for rec in git(*args).split(RS):
        rec = rec.strip("\n")
        if not rec:
            continue
        parts = rec.split(US)
        if len(parts) < 6:
            continue
        sha, an, ae, date, subject, body = parts[:6]
        if is_noise(an, ae, subject):
            continue
        items.append({"sha": sha, "author_date": date, "author": an,
                      "message": (subject + "\n\n" + body).strip(),
                      "origin": classify(ae, body)})
    return items


# Vibe commits are scarce & precious → scan the WHOLE tree (they touch ash/,
# login/, etc. outside include_dirs). Official commits are the bulk → restrict
# to include_dirs so the sample stays on code you care about.
vibe = [c for c in metadata_pass() if c["origin"] == "vibe"]
official = [c for c in metadata_pass(include_dirs) if c["origin"] == "official"]
print(f"After filtering: {len(vibe)} vibe (tree-wide), "
      f"{len(official)} official (in include_dirs)")

# ── 4a: FULL set, message-only -> commits_all.jsonl  (feeds the evolution RAG) ─
# RAG coverage is decoupled from the SFT cap: embedding messages is cheap, so we
# index EVERY (de-noised) commit's rationale, not just the sampled subset.
full = sorted(vibe + official, key=lambda c: c["author_date"])
all_path = data_path("commits_all.jsonl")
with open(all_path, "w") as out:
    for c in full:
        out.write(json.dumps({k: c[k] for k in
                  ("sha", "author_date", "author", "message", "origin")}) + "\n")
print(f"Wrote {len(full)} commits (message-only) -> {all_path}  [evolution RAG]")

# ── 4b: SAMPLED set with diffs -> commits.jsonl  (feeds the SFT dataset) ───────
selected = list(vibe) if prov["keep_all_vibe"] else []
cap = prov["max_official_commits"]
sampled_official = official
if cap and len(official) > cap:
    stride = len(official) / cap
    sampled_official = [official[int(i * stride)] for i in range(cap)]
    print(f"Sampled official down to {len(sampled_official)} (stride {stride:.1f}) for SFT")
selected += sampled_official
selected.sort(key=lambda c: c["author_date"])
print(f"Selected {len(selected)} commits for SFT (with diffs)")

# ── 5: fetch diffs only for the SFT-selected commits ─────────────────────────
out_path = data_path("commits.jsonl")
n = 0
with open(out_path, "w") as out:
    for c in selected:
        diff = git("show", c["sha"], "--no-merges", "--format=", "--unified=3",
                   "--", *ext_pathspecs)
        if not diff.strip():
            continue
        if len(diff) > max_diff:
            diff = diff[:max_diff] + "\n...[truncated]..."
        files = [l for l in git("show", c["sha"], "--name-only", "--format=").splitlines() if l.strip()][:50]
        out.write(json.dumps({**c, "files": files, "diff": diff}) + "\n")
        n += 1
        if n % 500 == 0:
            print(f"  diffed {n}/{len(selected)}...")

print(f"Wrote {n} commits -> {out_path}")
