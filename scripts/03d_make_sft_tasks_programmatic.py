"""Part 3D — Programmatic TASK->CODE pairs (free, bulk): teaches generation.

Flips the commit around: the subject becomes a developer *task*, the diff becomes
the target *implementation*. Classified by conventional-commit type so the
instruction matches (feat=implement, refactor=refactor, fix=fix, ...).

Output: data/sft_tasks_programmatic.jsonl   (assistant output = unified diff)
"""
import json
import re
from common import data_path

SYS = "You are a Chromium engineering assistant. Produce changes as unified diffs."
ORIGIN_LABEL = {"vibe": "vibe (company downstream)", "official": "upstream Chromium"}

# instruction templates by task type
VERB = {
    "feat":     "Implement the following feature in {origin}",
    "fix":      "Fix the following issue in {origin}",
    "refactor": "Refactor the following in {origin}",
    "perf":     "Improve performance as described in {origin}",
    "test":     "Add/adjust tests in {origin}",
    "build":    "Make the following build/config change in {origin}",
    "change":   "Make the following change in {origin}",
}


def task_type(subject):
    s = subject.lower().strip()
    m = re.match(r"^(feat|fix|refactor|perf|test|build|docs|style|chore)\b", s)
    if m:
        t = m.group(1)
        return {"docs": "change", "style": "refactor", "chore": "change"}.get(t, t)
    if s.startswith("revert"):
        return "fix"
    if "refactor" in s:
        return "refactor"
    if s.startswith(("add ", "implement", "introduce", "support ")):
        return "feat"
    if "fix" in s:
        return "fix"
    return "change"


def clean_subject(subject):
    # strip "feat(scope): " / "fix: " prefixes and trailing "(#1234)" PR refs
    s = re.sub(r"^(feat|fix|refactor|perf|test|build|docs|style|chore)(\([^)]*\))?:\s*", "", subject, flags=re.I)
    return re.sub(r"\s*\(#\d+\)\s*$", "", s).strip()


def pairs():
    for line in open(data_path("commits.jsonl")):
        c = json.loads(line)
        subject = c["message"].splitlines()[0]
        t = task_type(subject)
        origin = ORIGIN_LABEL.get(c.get("origin", "official"), "Chromium")
        verb = VERB.get(t, VERB["change"]).format(origin=origin)
        files = "\n".join(f"- {f}" for f in c.get("files", [])[:12])
        instruction = (f"{verb}:\n\n{clean_subject(subject)}\n\n"
                       f"Files to modify:\n{files}\n\n"
                       "Provide the change as a unified diff.")
        yield {"messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": f"```diff\n{c['diff']}\n```"},
        ]}


out_path = data_path("sft_tasks_programmatic.jsonl")
n = 0
with open(out_path, "w") as f:
    for ex in pairs():
        f.write(json.dumps(ex) + "\n")
        n += 1
print(f"Wrote {n} task->diff examples -> {out_path}")
