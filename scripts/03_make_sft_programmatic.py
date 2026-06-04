"""Part 3A — Free, mechanical SFT pairs derived from commits.

  Pair 1: diff -> commit message (ground-truth rationale, the "why")
  Pair 2: commit subject -> likely files touched

Output: data/sft_programmatic.jsonl
"""
import json
from common import data_path

SYS = "You are a Chromium engineering assistant."


ORIGIN_LABEL = {"vibe": "vibe downstream", "official": "upstream Chromium"}


def pairs():
    for line in open(data_path("commits.jsonl")):
        c = json.loads(line)
        msg = c["message"]
        subject = msg.splitlines()[0]
        kind = ORIGIN_LABEL.get(c.get("origin", "official"), "Chromium")

        yield {"messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content":
                f"Explain the intent of this {kind} change:\n\n```diff\n{c['diff'][:6000]}\n```"},
            {"role": "assistant", "content": msg},
        ]}

        if c["files"]:
            files = "\n".join(f"- {f}" for f in c["files"][:15])
            yield {"messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": f'Which files would a change titled "{subject}" likely touch?'},
                {"role": "assistant", "content": "Likely files:\n" + files},
            ]}


out_path = data_path("sft_programmatic.jsonl")
n = 0
with open(out_path, "w") as f:
    for ex in pairs():
        f.write(json.dumps(ex) + "\n")
        n += 1
print(f"Wrote {n} programmatic examples -> {out_path}")
