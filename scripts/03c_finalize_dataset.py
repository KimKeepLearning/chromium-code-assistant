"""Part 3C — Merge + dedup + per-source cap + split the SFT sources.

Caps (dataset.source_caps in config) balance the mix so the bulky FREE generators
don't swamp the scarce TEACHER pairs. Capping is applied per source, after dedup,
with a fixed seed (reproducible).

Outputs: data/train.json, data/eval.json
"""
import hashlib
import json
import os
import random
from common import load_config, data_path

cfg = load_config()
caps = cfg["dataset"].get("source_caps") or {}

# (file, category) — category lets us report the why/generation balance.
SOURCES = [
    ("sft_programmatic.jsonl",       "why·free"),       # 03
    ("sft_teacher.jsonl",            "why·teacher"),     # 03b
    ("sft_tasks_programmatic.jsonl", "gen·free"),        # 03d
    ("sft_tasks_teacher.jsonl",      "gen·teacher"),     # 03e
]

random.seed(0)
seen = set()
rows = []
cat_counts = {}
for fn, cat in SOURCES:
    path = data_path(fn)
    if not os.path.exists(path):
        print(f"  (skip missing {fn})")
        continue
    src_rows = []
    for line in open(path):
        h = hashlib.md5(line.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        src_rows.append(json.loads(line))
    loaded = len(src_rows)
    cap = caps.get(fn)
    if cap and loaded > cap:
        random.shuffle(src_rows)          # cap a representative random subset
        src_rows = src_rows[:cap]
    print(f"  {fn:32} {loaded:6} loaded -> {len(src_rows):6} kept  (cap {cap if cap else '-'})")
    rows.extend(src_rows)
    cat_counts[cat] = cat_counts.get(cat, 0) + len(src_rows)

random.shuffle(rows)
n = int(len(rows) * cfg["dataset"]["train_split"])
json.dump(rows[:n], open(data_path("train.json"), "w"))
json.dump(rows[n:], open(data_path("eval.json"), "w"))

why = cat_counts.get("why·free", 0) + cat_counts.get("why·teacher", 0)
gen = cat_counts.get("gen·free", 0) + cat_counts.get("gen·teacher", 0)
teach = cat_counts.get("why·teacher", 0) + cat_counts.get("gen·teacher", 0)
total = len(rows) or 1
print(f"\nBalance: why={why} ({why*100//total}%)  generation={gen} ({gen*100//total}%)  "
      f"| teacher={teach} ({teach*100//total}%)")
print(f"Total {len(rows)} examples -> {n} train / {len(rows) - n} eval")
