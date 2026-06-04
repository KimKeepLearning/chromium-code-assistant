"""Part 2B — Extract commit messages + diffs for the range versions.from..to.

Output: data/commits.jsonl  (one commit per line: sha, date, message, files, diff)
This file feeds BOTH the SFT dataset (Part 3) and the RAG 'evolution' index (Part 5).
"""
import json
from git import Repo
from common import load_config, data_path

cfg = load_config()
repo = Repo(cfg["repo"]["path"])
rng = f"{cfg['versions']['from']}..{cfg['versions']['to']}"
exts = tuple(cfg["extensions"])
pathspecs = [f"*{e}" for e in exts]
max_diff = cfg["dataset"]["max_diff_chars"]

out_path = data_path("commits.jsonl")
n = 0
with open(out_path, "w") as out:
    for c in repo.iter_commits(rng, no_merges=True):
        try:
            diff = repo.git.diff(f"{c.hexsha}~1", c.hexsha, "--unified=3", "--", *pathspecs)
        except Exception:
            diff = ""
        if not diff:
            continue
        if len(diff) > max_diff:
            diff = diff[:max_diff] + "\n...[truncated]..."
        out.write(json.dumps({
            "sha": c.hexsha,
            "author_date": c.authored_datetime.isoformat(),
            "message": c.message.strip(),
            "files": list(c.stats.files.keys())[:50],
            "diff": diff,
        }) + "\n")
        n += 1
        if n % 500 == 0:
            print(f"  {n} commits...")

print(f"Wrote {n} commits -> {out_path}")
