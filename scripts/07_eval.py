"""Part 7 — Quick smoke eval against the running RAG proxy.

Provide data/eval_questions.json:  [{"q": "...", "expect_substr": "..."}, ...]
(e.g. a known file path, symbol, or commit sha you expect in a good answer)
"""
import json
import os
import httpx
from common import load_config, data_path

cfg = load_config()
url = f"http://localhost:{cfg['serve']['proxy_port']}/v1/chat/completions"
qfile = data_path("eval_questions.json")

if not os.path.exists(qfile):
    raise SystemExit(f"Create {qfile} first: a list of {{q, expect_substr}} objects.")

qs = json.load(open(qfile))
hit = 0
for ex in qs:
    r = httpx.post(url, json={"messages": [{"role": "user", "content": ex["q"]}]},
                   timeout=180).json()
    ans = r["choices"][0]["message"]["content"]
    ok = ex["expect_substr"].lower() in ans.lower()
    hit += ok
    print(f"[{'OK ' if ok else 'MISS'}] {ex['q'][:70]}")

print(f"\n{hit}/{len(qs)} answers contained the expected reference")
