"""Part 3E — Teacher TASK->CODE pairs: natural "develop/refactor" examples.

Turns each commit into ONE realistic developer request + an ideal answer that
explains the approach and shows full code (not a raw diff). Prioritizes
feat/refactor/fix commits and vibe (downstream) ones — the generation signal.

Output: data/sft_tasks_teacher.jsonl
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

PROMPT = """You are creating training data that teaches a model to DEVELOP and
REFACTOR Chromium code in the project's own style. Given this real {kind} commit
(message + diff), write ONE realistic developer interaction:
- "q": a natural task a developer would type (e.g. "Add ...", "Refactor ...",
  "Fix ..."), described from intent WITHOUT revealing the exact solution/diff.
- "a": an ideal assistant answer that briefly explains the approach, then shows
  the key code as full functions/snippets (with file paths), matching how this
  commit actually solved it. Prefer real Chromium idioms.
Return ONLY JSON: {{"q": "...", "a": "..."}}.

COMMIT MESSAGE:
{msg}

DIFF (truncated):
{diff}
"""

ORIGIN_LABEL = {"vibe": "vibe downstream", "official": "upstream Chromium"}
SYS = "You are a Chromium engineering assistant."
out_path = data_path("sft_tasks_teacher.jsonl")

GEN_TYPES = ("feat", "fix", "refactor", "perf")


def is_generation_commit(c):
    s = c["message"].splitlines()[0].lower()
    return s.startswith(GEN_TYPES) or any(k in s for k in
        ("add ", "implement", "introduce", "refactor", "fix", "support "))


def extract_obj(content):
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", content).strip()
    try:
        o = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None
        o = json.loads(m.group(0))
    return o if isinstance(o, dict) and "q" in o and "a" in o else None


def distill(c):
    r = client.chat.completions.create(
        model=dcfg["teacher_model"],
        messages=[{"role": "user", "content": PROMPT.format(
            kind=ORIGIN_LABEL.get(c.get("origin", "official"), "Chromium"),
            msg=c["message"][:3000], diff=c["diff"][:6000])}],
        temperature=0.5,
        timeout=60,
    )
    o = extract_obj(r.choices[0].message.content)
    if not o:
        return []
    return [{"messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": o["q"]},
        {"role": "assistant", "content": o["a"]},
    ]}]


# Prioritize generation-type commits, and vibe over official within that.
commits = [json.loads(l) for l in open(data_path("commits.jsonl"))]
commits.sort(key=lambda c: (0 if is_generation_commit(c) else 1,
                            0 if c.get("origin") == "vibe" else 1))
commits = commits[:dcfg.get("max_task_teacher_commits", 1500)]
workers = dcfg.get("teacher_concurrency", 1)
print(f"Distilling {len(commits)} task->code commits "
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
                for row in fut.result():
                    out.write(json.dumps(row) + "\n")
                    n_pairs += 1
                out.flush()
            except Exception as e:
                n_err += 1
                if n_err <= 3:
                    print(f"  [err @ commit {i}] {type(e).__name__}: {e}", flush=True)
            if n_done % 20 == 0 or n_done == len(commits):
                print(f"  {n_done}/{len(commits)} done | {n_pairs} pairs | {n_err} errors",
                      flush=True)

print(f"Wrote {n_pairs} task->code pairs ({n_err} errors) -> {out_path}")
