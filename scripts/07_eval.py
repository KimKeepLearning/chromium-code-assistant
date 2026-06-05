"""Part 7 — A/B eval: base+RAG vs ft+RAG vs frontier+RAG.

Retrieves the SAME RAG context per question, then sends it to each system in
config.eval.systems. This isolates *model quality given identical context* — the
number you need to decide whether the fine-tune earns its keep over base+RAG.

  data/eval_questions.json : [{"q": "...", "expect_substr": "..."}, ...]

Modes:
  python 07_eval.py            # substring hit-rate (cheap, deterministic)
  python 07_eval.py --judge    # + LLM-judge: grade each answer 1-5 on
                               #   correctness (vs retrieved context) and idiom

Writes data/eval_results.json with every answer (and scores) for manual reading.

NOTE: "frontier+RAG" is a strong model with the same single-shot RAG context, NOT a
tool-using agent. A real agent (live repo + git) could do better; this measures the
model, holding retrieval constant.
"""
import argparse
import json
import os
import re
import time
import chromadb
import httpx
from sentence_transformers import SentenceTransformer
from common import load_config, data_path, root_path

ap = argparse.ArgumentParser()
ap.add_argument("--judge", action="store_true", help="grade answers 1-5 with an LLM judge")
args = ap.parse_args()

cfg = load_config()
rcfg = cfg["rag"]
ecfg = cfg["eval"]
systems = ecfg["systems"]

qfile = data_path("eval_questions.json")
if not os.path.exists(qfile):
    raise SystemExit(f"Create {qfile} first: a list of {{q, expect_substr}} objects "
                     f"(see data/eval_questions.example.json).")
questions = json.load(open(qfile))

emb = SentenceTransformer(rcfg["embed_model"])
db = chromadb.PersistentClient(path=root_path(rcfg["db_path"]))
code = db.get_collection(rcfg["code_collection"])
evo = db.get_collection(rcfg["evolution_collection"])


def retrieve(q):
    def hit(coll, k):
        r = coll.query(query_embeddings=emb.encode([q], normalize_embeddings=True).tolist(),
                       n_results=k)
        return list(zip(r["documents"][0], r["metadatas"][0]))
    ctx = "## Current code (R148)\n" + "\n\n".join(
        f"// {m['path']}\n{d}" for d, m in hit(code, rcfg["top_k_code"]))
    ctx += "\n\n## Evolution (commits R136 -> R148)\n" + "\n\n".join(
        d for d, _ in hit(evo, rcfg["top_k_evolution"]))
    return ctx


SYS = ("You are a Chromium engineering assistant. Use the retrieved context as "
       "ground truth for current code and rationale. Cite file paths and commit "
       "shas. If context is insufficient, say so.\n\n")


def chat(base_url, model, messages, api_key_env=None, temperature=0.2):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {}
    key = os.environ.get(api_key_env) if api_key_env else None
    if key:
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.time()
    resp = httpx.post(url, json={"model": model, "messages": messages,
                                 "temperature": temperature, "stream": False},
                      headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"], time.time() - t0


def ask(system, messages):
    return chat(system["base_url"], system["model"], messages,
                system.get("api_key_env"))


# ── LLM judge ────────────────────────────────────────────────────────────────
JUDGE_PROMPT = """You are grading an AI assistant's answer to a Chromium
engineering question. Treat the REFERENCE CONTEXT as ground truth. Grade two axes,
each an integer 1-5:
- CORRECTNESS: technical accuracy vs the context (5=fully correct, 1=wrong/hallucinated).
- IDIOM: how much it reads like real Chromium engineering — terminology, conventions,
  concrete file/symbol references (5=native, 1=generic).

Output EXACTLY this, nothing else:
CORRECTNESS: <n>
IDIOM: <n>
REASON: <one short line>

QUESTION:
{q}

REFERENCE CONTEXT:
{ctx}

ANSWER TO GRADE:
{ans}
"""


def _first_int(s):
    m = re.search(r"[1-5]", s)
    return int(m.group(0)) if m else None


def judge(q, ctx, ans):
    jc = ecfg["judge"]
    text, _ = chat(jc["base_url"], jc["model"],
                   [{"role": "user", "content": JUDGE_PROMPT.format(
                       q=q, ctx=ctx[:4000], ans=ans[:4000])}],
                   jc.get("api_key_env"), temperature=0.0)
    corr = idiom = None
    reason = ""
    for line in text.splitlines():
        u = line.strip().upper()
        if u.startswith("CORRECTNESS:"):
            corr = _first_int(line)
        elif u.startswith("IDIOM:"):
            idiom = _first_int(line)
        elif u.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return corr, idiom, reason


# ── probe reachable systems ──────────────────────────────────────────────────
live = []
for s in systems:
    try:
        ask(s, [{"role": "user", "content": "ping"}])
        live.append(s)
        print(f"  [up]   {s['name']} ({s['model']})")
    except Exception as e:
        print(f"  [skip] {s['name']} ({s['model']}): {type(e).__name__}")
if not live:
    raise SystemExit("No eval systems reachable — start Ollama/vLLM or set the API key.")
if args.judge:
    print(f"  [judge] {ecfg['judge']['model']} via {ecfg['judge']['base_url']}")

# ── run ──────────────────────────────────────────────────────────────────────
stats = {s["name"]: {"hits": 0, "errors": 0, "lat": 0.0,
                     "corr": 0, "idiom": 0, "judged": 0} for s in live}
results = []
for ex in questions:
    ctx = retrieve(ex["q"])
    messages = [{"role": "system", "content": SYS + ctx},
                {"role": "user", "content": ex["q"]}]
    row = {"q": ex["q"], "expect_substr": ex.get("expect_substr", ""), "answers": {}}
    for s in live:
        try:
            ans, lat = ask(s, messages)
            hit = ex.get("expect_substr", "").lower() in ans.lower()
            stats[s["name"]]["hits"] += hit
            stats[s["name"]]["lat"] += lat
            entry = {"hit": hit, "latency": round(lat, 1), "text": ans}
            if args.judge:
                try:
                    corr, idiom, reason = judge(ex["q"], ctx, ans)
                    entry["correctness"], entry["idiom"], entry["reason"] = corr, idiom, reason
                    if corr:
                        stats[s["name"]]["corr"] += corr
                        stats[s["name"]]["judged"] += 1
                    if idiom:
                        stats[s["name"]]["idiom"] += idiom
                except Exception as e:
                    entry["judge_error"] = f"{type(e).__name__}: {e}"
            row["answers"][s["name"]] = entry
        except Exception as e:
            stats[s["name"]]["errors"] += 1
            row["answers"][s["name"]] = {"error": f"{type(e).__name__}: {e}"}
    results.append(row)
    mark = "".join("✓" if row["answers"][s["name"]].get("hit") else "·" for s in live)
    print(f"  [{mark}] {ex['q'][:60]}")

json.dump(results, open(data_path("eval_results.json"), "w"), indent=2)

# ── report ───────────────────────────────────────────────────────────────────
n = len(questions)
if args.judge:
    print(f"\n{'system':16} {'hits':>10} {'correct':>9} {'idiom':>7} {'lat':>8}  err")
    for s in live:
        st = stats[s["name"]]
        j = max(st["judged"], 1)
        done = max(n - st["errors"], 1)
        print(f"{s['name']:16} {st['hits']:>4}/{n:<5} {st['corr']/j:>8.2f} "
              f"{st['idiom']/j:>7.2f} {st['lat']/done:>6.1f}s  {st['errors']}")
    print("\n(correct/idiom are 1-5 LLM-judge averages over judged answers)")
else:
    print(f"\n{'system':16} {'hits':>10} {'avg latency':>13}  errors")
    for s in live:
        st = stats[s["name"]]
        done = max(n - st["errors"], 1)
        print(f"{s['name']:16} {st['hits']:>4}/{n:<5} {st['lat']/done:>11.1f}s  {st['errors']}")
    print("\nTip: add --judge for 1-5 correctness/idiom grading (more meaningful than hits).")
print(f"Full answers -> {data_path('eval_results.json')}")
