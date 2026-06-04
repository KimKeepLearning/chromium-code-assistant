"""Part 7 — A/B eval: base+RAG vs ft+RAG vs frontier+RAG.

Retrieves the SAME RAG context per question, then sends it to each system in
config.eval.systems. This isolates *model quality given identical context* — the
number you need to decide whether the fine-tune earns its keep over base+RAG.

  data/eval_questions.json : [{"q": "...", "expect_substr": "..."}, ...]
  -> prints a comparison table (substring-hit rate + latency)
  -> writes data/eval_results.json with every answer for manual reading

NOTE: "frontier+RAG" here is a strong model with the same single-shot RAG context,
NOT a tool-using agent. A real agent (live repo + git access) could do better; this
measures the model, holding retrieval constant.
"""
import json
import os
import time
import chromadb
import httpx
from sentence_transformers import SentenceTransformer
from common import load_config, data_path, root_path

cfg = load_config()
rcfg = cfg["rag"]
systems = cfg["eval"]["systems"]

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


def ask(system, messages):
    url = system["base_url"].rstrip("/") + "/chat/completions"
    headers = {}
    key = os.environ.get(system["api_key_env"]) if system.get("api_key_env") else None
    if key:
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.time()
    resp = httpx.post(url, json={"model": system["model"], "messages": messages,
                                 "temperature": 0.2, "stream": False},
                      headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"], time.time() - t0


# Probe which systems are reachable, so we only compare what's actually up.
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

stats = {s["name"]: {"hits": 0, "errors": 0, "lat": 0.0} for s in live}
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
            row["answers"][s["name"]] = {"hit": hit, "latency": round(lat, 1), "text": ans}
        except Exception as e:
            stats[s["name"]]["errors"] += 1
            row["answers"][s["name"]] = {"error": f"{type(e).__name__}: {e}"}
    results.append(row)
    mark = "".join("✓" if row["answers"][s["name"]].get("hit") else "·" for s in live)
    print(f"  [{mark}] {ex['q'][:60]}")

json.dump(results, open(data_path("eval_results.json"), "w"), indent=2)

n = len(questions)
print(f"\n{'system':16} {'hits':>10} {'avg latency':>13}  errors")
for s in live:
    st = stats[s["name"]]
    done = max(n - st["errors"], 1)
    print(f"{s['name']:16} {st['hits']:>4}/{n:<5} {st['lat']/done:>11.1f}s  {st['errors']}")
print(f"\nFull answers -> {data_path('eval_results.json')}  (read these — "
      f"substring hits undersell quality differences)")
