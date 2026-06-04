"""Part 3B — Teacher-distilled SFT pairs (the high-quality "why"/architecture set).

Uses any OpenAI-compatible API. Defaults to DeepSeek (China-friendly billing):
  config.yaml -> dataset.teacher_base_url / teacher_model
  export OPENAI_API_KEY=<your DeepSeek key>

Output: data/sft_teacher.jsonl
"""
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def extract_pairs(content):
    """Robustly pull the JSON list of {q,a} out of a model reply."""
    content = content.strip()
    if content.startswith("```"):                 # strip ``` / ```json fences
        content = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", content).strip()
    try:
        data = json.loads(content)
    except Exception:
        m = re.search(r"\[.*\]", content, re.DOTALL)   # grab first [...] block
        if not m:
            return []
        data = json.loads(m.group(0))
    return [p for p in data if isinstance(p, dict) and "q" in p and "a" in p]


def distill(c):
    """One commit -> list of chat examples (raises on API error for the caller)."""
    r = client.chat.completions.create(
        model=dcfg["teacher_model"],
        messages=[{"role": "user", "content": PROMPT.format(
            kind=ORIGIN_LABEL.get(c.get("origin", "official"), "Chromium"),
            msg=c["message"][:3000], diff=c["diff"][:6000])}],
        temperature=0.4,
        timeout=60,
    )
    return [{"messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": p["q"]},
        {"role": "assistant", "content": p["a"]},
    ]} for p in extract_pairs(r.choices[0].message.content)]


# Prioritize vibe (downstream) commits — the scarce company signal — then official.
commits = [json.loads(l) for l in open(data_path("commits.jsonl"))]
commits.sort(key=lambda c: 0 if c.get("origin") == "vibe" else 1)
commits = commits[:dcfg["max_teacher_commits"]]
workers = dcfg.get("teacher_concurrency", 1)
print(f"Distilling {len(commits)} commits "
      f"({sum(c.get('origin') == 'vibe' for c in commits)} vibe-prioritized) "
      f"with {workers} workers")

n_done = n_pairs = n_err = 0
lock = threading.Lock()
with open(out_path, "w") as out, ThreadPoolExecutor(max_workers=workers) as ex:
    futures = {ex.submit(distill, c): i for i, c in enumerate(commits)}
    for fut in as_completed(futures):
        i = futures[fut]
        with lock:
            n_done += 1
            try:
                for ex_row in fut.result():
                    out.write(json.dumps(ex_row) + "\n")
                    n_pairs += 1
                out.flush()
            except Exception as e:
                n_err += 1
                if n_err <= 3:
                    print(f"  [err @ commit {i}] {type(e).__name__}: {e}", flush=True)
            if n_done % 20 == 0 or n_done == len(commits):
                print(f"  {n_done}/{len(commits)} done | {n_pairs} pairs | {n_err} errors",
                      flush=True)

print(f"Wrote {n_pairs} teacher pairs ({n_err} errors) -> {out_path}")
