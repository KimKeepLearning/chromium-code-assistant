"""Part 5 — Build/update the RAG index: two collections in one Chroma DB.

  code_<label_to> : current source chunks       (the FACTS)
  evolution       : selected commits, vibe+official, R136..R148 (the WHY)

Modes:
  python 05_build_rag.py                 # full (re)build — drops & rebuilds both
  python 05_build_rag.py --evolution     # rebuild only the evolution collection
  python 05_build_rag.py --code          # rebuild only the code collection
  python 05_build_rag.py --changed-since <ref>
        # incremental: only re-embed code files changed between <ref> and HEAD
        # (use after a git pull/rebase — fast). e.g. --changed-since HEAD@{1}

Each chunk/commit carries an `origin` ('code' / 'vibe' / 'official') so the
proxy and you can tell company customizations from upstream.
"""
import argparse
import glob
import json
import os
import subprocess
import chromadb
from sentence_transformers import SentenceTransformer
from common import load_config, data_path, root_path

cfg = load_config()
rcfg, v = cfg["rag"], cfg["versions"]
CHROMIUM = cfg["repo"]["path"]
EXT = tuple(cfg["extensions"])
CHUNK, OVERLAP = rcfg["chunk_size"], rcfg["chunk_overlap"]
CODE_COLL = rcfg["code_collection"]

emb = SentenceTransformer(rcfg["embed_model"])
client = chromadb.PersistentClient(path=root_path(rcfg["db_path"]))


def embed(texts):
    return emb.encode(texts, normalize_embeddings=True).tolist()


def git(*args):
    return subprocess.run(["git", "-C", CHROMIUM, *args],
                          capture_output=True, text=True, errors="ignore").stdout


def chunks_of(src):
    step = CHUNK - OVERLAP
    for i in range(0, max(len(src), 1), step):
        c = src[i:i + CHUNK]
        if c.strip():
            yield i, c


def index_file(coll, abspath):
    """(Re)index a single file: drop its old chunks, add fresh ones."""
    rel = os.path.relpath(abspath, CHROMIUM)
    coll.delete(where={"path": rel})
    try:
        src = open(abspath, encoding="utf-8", errors="ignore").read()
    except Exception:
        return 0
    ids, docs, meta = [], [], []
    for i, chunk in chunks_of(src):
        ids.append(f"{rel}:{i}")
        docs.append(chunk)
        meta.append({"path": rel, "kind": "code", "origin": "code", "version": v["label_to"]})
    if docs:
        coll.add(ids=ids, documents=docs, embeddings=embed(docs), metadatas=meta)
    return len(docs)


def build_code_full():
    if CODE_COLL in [c.name for c in client.list_collections()]:
        client.delete_collection(CODE_COLL)
    coll = client.get_or_create_collection(CODE_COLL)
    ids, docs, meta = [], [], []
    for d in cfg["include_dirs"]:
        for path in glob.glob(f"{CHROMIUM}/{d}/**/*", recursive=True):
            if not path.endswith(EXT) or not os.path.isfile(path):
                continue
            rel = os.path.relpath(path, CHROMIUM)
            try:
                src = open(path, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            for i, chunk in chunks_of(src):
                ids.append(f"{rel}:{i}")
                docs.append(chunk)
                meta.append({"path": rel, "kind": "code", "origin": "code", "version": v["label_to"]})
            if len(docs) >= 256:
                coll.add(ids=ids, documents=docs, embeddings=embed(docs), metadatas=meta)
                ids, docs, meta = [], [], []
    if docs:
        coll.add(ids=ids, documents=docs, embeddings=embed(docs), metadatas=meta)
    print(f"code collection: {coll.count()} chunks (full build)")


def build_code_incremental(since_ref):
    coll = client.get_or_create_collection(CODE_COLL)
    in_scope = tuple(f"{d}/" for d in cfg["include_dirs"])
    changed = git("diff", "--name-status", since_ref, "HEAD").splitlines()
    n_idx = n_del = 0
    for line in changed:
        parts = line.split("\t")
        status, rel = parts[0], parts[-1]
        if not rel.startswith(in_scope) or not rel.endswith(EXT):
            continue
        if status.startswith("D"):
            coll.delete(where={"path": rel})
            n_del += 1
        else:
            index_file(coll, os.path.join(CHROMIUM, rel))
            n_idx += 1
    print(f"code collection: re-indexed {n_idx} files, removed {n_del} "
          f"(changed since {since_ref}); total {coll.count()} chunks")


def build_evolution():
    name = rcfg["evolution_collection"]
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    coll = client.get_or_create_collection(name)
    # Prefer the FULL message-only set (all commits) over the sampled SFT set.
    commits_file = data_path("commits_all.jsonl")
    if not os.path.exists(commits_file):
        commits_file = data_path("commits.jsonl")
    if not os.path.exists(commits_file):
        print("  (no commits_all.jsonl / commits.jsonl — run 02_extract_commits.py first)")
        return
    print(f"  indexing evolution from {os.path.basename(commits_file)}")
    ids, docs, meta = [], [], []
    for line in open(commits_file):
        c = json.loads(line)
        tag = f"[{c.get('origin', 'official')}]"
        doc = (f"{tag} [{c['author_date']}] {c['message']}\n\n"
               f"Files: {', '.join(c.get('files', [])[:20])}")
        ids.append(c["sha"])
        docs.append(doc)
        meta.append({"kind": "commit", "origin": c.get("origin", "official"),
                     "sha": c["sha"], "date": c["author_date"]})
        if len(docs) >= 256:
            coll.add(ids=ids, documents=docs, embeddings=embed(docs), metadatas=meta)
            ids, docs, meta = [], [], []
    if docs:
        coll.add(ids=ids, documents=docs, embeddings=embed(docs), metadatas=meta)
    print(f"evolution collection: {coll.count()} commits")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", action="store_true", help="rebuild code collection only")
    ap.add_argument("--evolution", action="store_true", help="rebuild evolution collection only")
    ap.add_argument("--changed-since", metavar="REF",
                    help="incremental code re-index of files changed REF..HEAD")
    args = ap.parse_args()

    do_code = args.code or args.changed_since or not (args.code or args.evolution)
    do_evo = args.evolution or not (args.code or args.evolution or args.changed_since)

    if do_code:
        if args.changed_since:
            build_code_incremental(args.changed_since)
        else:
            build_code_full()
    if do_evo:
        build_evolution()
    print("RAG update done.")
