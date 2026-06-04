"""Part 3C — Merge + dedup + split the SFT sources into train/eval.

Inputs : data/sft_programmatic.jsonl, data/sft_teacher.jsonl (either may be absent)
Outputs: data/train.json, data/eval.json
"""
import hashlib
import json
import os
import random
from common import load_config, data_path

cfg = load_config()
sources = ["sft_programmatic.jsonl", "sft_teacher.jsonl"]

seen, rows = set(), []
for fn in sources:
    path = data_path(fn)
    if not os.path.exists(path):
        print(f"  (skip missing {fn})")
        continue
    for line in open(path):
        h = hashlib.md5(line.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        rows.append(json.loads(line))

random.seed(0)
random.shuffle(rows)
n = int(len(rows) * cfg["dataset"]["train_split"])
json.dump(rows[:n], open(data_path("train.json"), "w"))
json.dump(rows[n:], open(data_path("eval.json"), "w"))
print(f"Total {len(rows)} examples -> {n} train / {len(rows) - n} eval")
