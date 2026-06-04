"""Part 5 — Build the RAG index: two collections in one Chroma DB.

  code_r148  : current source chunks (the FACTS)
  evolution  : commit messages+files R136..R148 (the WHY)

Re-run anytime you pull new code — this is how knowledge stays fresh without
retraining. Indexing only changed dirs is fine (edit include_dirs).
"""
import glob
import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from common import load_config, data_path

cfg = load_config()
rcfg = cfg["rag"]
CHROMIUM = cfg["repo"]["path"]
EXT = tuple(cfg["extensions"])
CHUNK, OVERLAP = rcfg["chunk_size"], rcfg["chunk_overlap"]

emb = SentenceTransformer(rcfg["embed_model"])
client = chromadb.PersistentClient(path=os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rcfg["db_path"]))


def embed(texts):
    return emb.encode(texts, normalize_embeddings=True).tolist()


def flush(coll, ids, docs, meta):
    if docs:
        coll.add(ids=ids, documents=docs, embeddings=embed(docs), metadatas=meta)


# ── Collection 1: current code ──────────────────────────────────────────────
code = client.get_or_create_collection(rcfg["code_collection"])
ids, docs, meta = [], [], []
ver = cfg["versions"]["to"]
for d in cfg["include_dirs"]:
    for path in glob.glob(f"{CHROMIUM}/{d}/**/*", recursive=True):
        if not path.endswith(EXT) or not os.path.isfile(path):
            continue
        try:
            src = open(path, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        rel = os.path.relpath(path, CHROMIUM)
        step = CHUNK - OVERLAP
        for i in range(0, max(len(src), 1), step):
            chunk = src[i:i + CHUNK]
            if not chunk.strip():
                continue
            ids.append(f"{rel}:{i}")
            docs.append(chunk)
            meta.append({"path": rel, "kind": "code", "version": ver})
        if len(docs) >= 256:
            flush(code, ids, docs, meta)
            ids, docs, meta = [], [], []
flush(code, ids, docs, meta)
print(f"code collection: {code.count()} chunks")

# ── Collection 2: evolution / WHY ───────────────────────────────────────────
evo = client.get_or_create_collection(rcfg["evolution_collection"])
ids, docs, meta = [], [], []
commits_file = data_path("commits.jsonl")
if os.path.exists(commits_file):
    for line in open(commits_file):
        c = json.loads(line)
        doc = f"[{c['author_date']}] {c['message']}\n\nFiles: {', '.join(c['files'][:20])}"
        ids.append(c["sha"])
        docs.append(doc)
        meta.append({"kind": "commit", "sha": c["sha"], "date": c["author_date"]})
        if len(docs) >= 256:
            flush(evo, ids, docs, meta)
            ids, docs, meta = [], [], []
    flush(evo, ids, docs, meta)
print(f"evolution collection: {evo.count()} commits")
print("RAG built.")
