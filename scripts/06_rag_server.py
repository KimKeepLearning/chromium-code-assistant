"""Part 6B — RAG proxy: OpenAI-compatible endpoint that does retrieve->augment->generate.

Point VS Code (Continue) at  http://localhost:<proxy_port>/v1
It forwards to your fine-tuned model (Ollama/vLLM, set serve.model_url).

Run:
  uvicorn 06_rag_server:app --port 9000      # from the scripts/ dir
  # or: python scripts/06_rag_server.py
"""
import os
import chromadb
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from common import load_config

cfg = load_config()
rcfg, scfg = cfg["rag"], cfg["serve"]

emb = SentenceTransformer(rcfg["embed_model"])
db = chromadb.PersistentClient(path=os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rcfg["db_path"]))
code = db.get_collection(rcfg["code_collection"])
evo = db.get_collection(rcfg["evolution_collection"])

app = FastAPI()


class Req(BaseModel):
    messages: list
    temperature: float = 0.2
    stream: bool = False


def retrieve(q, coll, k):
    r = coll.query(
        query_embeddings=emb.encode([q], normalize_embeddings=True).tolist(),
        n_results=k)
    return list(zip(r["documents"][0], r["metadatas"][0]))


@app.post("/v1/chat/completions")
async def chat(req: Req):
    user_q = next((m["content"] for m in reversed(req.messages)
                   if m["role"] == "user"), "")
    code_hits = retrieve(user_q, code, rcfg["top_k_code"])
    evo_hits = retrieve(user_q, evo, rcfg["top_k_evolution"])

    ctx = "## Current code (R148)\n" + "\n\n".join(
        f"// {m['path']}\n{d}" for d, m in code_hits)
    ctx += "\n\n## Evolution (commits R136 -> R148)\n" + "\n\n".join(
        d for d, _ in evo_hits)

    sys = ("You are a Chromium engineering assistant. Use the retrieved context "
           "as ground truth for current code and rationale. Cite file paths and "
           "commit shas. If context is insufficient, say so explicitly.\n\n" + ctx)

    payload = {
        "model": scfg["model_name"],
        "messages": [{"role": "system", "content": sys}, *req.messages],
        "temperature": req.temperature,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=180) as cx:
        resp = await cx.post(scfg["model_url"], json=payload)
    return resp.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=cfg["serve"]["proxy_port"])
