"""Part 3B — Teacher-distilled SFT pairs (the high-quality "why"/architecture set).

Uses an OpenAI-compatible API. Set OPENAI_API_KEY (and optionally
dataset.teacher_base_url in config.yaml to point at a Claude-compatible endpoint).

Output: data/sft_teacher.jsonl
"""
import json
import os
from openai import OpenAI
from common import load_config, data_path

cfg = load_config()
dcfg = cfg["dataset"]
client = OpenAI(base_url=dcfg.get("teacher_base_url") or None)

PROMPT = """You are creating training data for a Chromium coding assistant.
Given this real {kind} commit (message + diff), produce 2 diverse Q&A pairs a
Chromium engineer might ask. Focus on WHY the change was made, the architecture
involved, and how it relates to evolution between Chrome milestones (R136 -> R148).
If this is a vibe downstream change, make clear it is a company-specific
customization layered on upstream Chromium.
Return ONLY a JSON list: [{{"q": "...", "a": "..."}}].

COMMIT MESSAGE:
{msg}

DIFF (truncated):
{diff}
"""

ORIGIN_LABEL = {"vibe": "vibe downstream", "official": "upstream Chromium"}

SYS = "You are a Chromium engineering assistant."
out_path = data_path("sft_teacher.jsonl")

# Distillation is the expensive part — prioritize vibe (downstream) commits, the
# scarce company signal, then fill the budget with official ones.
commits = [json.loads(l) for l in open(data_path("commits.jsonl"))]
commits.sort(key=lambda c: 0 if c.get("origin") == "vibe" else 1)
commits = commits[:dcfg["max_teacher_commits"]]
print(f"Distilling {len(commits)} commits "
      f"({sum(c.get('origin') == 'vibe' for c in commits)} vibe-prioritized)")

n_pairs = 0
with open(out_path, "w") as out:
    for i, c in enumerate(commits):
        try:
            r = client.chat.completions.create(
                model=dcfg["teacher_model"],
                messages=[{"role": "user", "content": PROMPT.format(
                    kind=ORIGIN_LABEL.get(c.get("origin", "official"), "Chromium"),
                    msg=c["message"][:3000], diff=c["diff"][:6000])}],
                temperature=0.4,
            )
            content = r.choices[0].message.content.strip()
            # tolerate ```json fences
            if content.startswith("```"):
                content = content.split("```")[1].lstrip("json").strip()
            for p in json.loads(content):
                out.write(json.dumps({"messages": [
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": p["q"]},
                    {"role": "assistant", "content": p["a"]},
                ]}) + "\n")
                n_pairs += 1
        except Exception as e:
            continue
        if i % 200 == 0:
            print(f"  processed {i} commits, {n_pairs} pairs...")

print(f"Wrote {n_pairs} teacher pairs -> {out_path}")
