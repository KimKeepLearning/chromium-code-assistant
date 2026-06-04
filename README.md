# Chromium Code Assistant

A Chromium-specialized coding assistant that combines **RAG** (for current R148
code and the *why* behind R136→R148 evolution) with a **QLoRA fine-tune** (for
Chromium idioms, conventions, and reasoning style).

> **Core principle:** fine-tuning adapts *behavior/style*; RAG holds the *facts*.
> Don't try to bake current Chromium code into the weights — it's too big, changes
> daily, and you'd get a confident, stale, hallucinating model. Keep the split.

Target hardware: **NVIDIA RTX 4070 Ti (16GB)**, Windows + **WSL2** (Ubuntu) for training.

```
chromium repo (R148 + history) ──► extract code + commits ──► RAG index (facts + why)
                                          │
                                          └► build SFT dataset ──► QLoRA fine-tune
                                                                        │
                          serve: Ollama/vLLM + RAG proxy ◄──────────────┘
                                          │
                                   VS Code (Continue)
```

## Layout
| Path | Purpose |
|---|---|
| `config.yaml` | Single source of truth — paths, version anchors, hyperparams. Edit first. |
| `scripts/common.py` | Config loader shared by all scripts. |
| `scripts/01_setup_repo.sh` | Fetch mirror + milestone history. |
| `scripts/02_extract_commits.py` | Commit msgs + diffs (R136→R148) → `data/commits.jsonl`. |
| `scripts/03_make_sft_programmatic.py` | Free mechanical SFT pairs. |
| `scripts/03b_make_sft_teacher.py` | Teacher-distilled SFT pairs (needs API key). |
| `scripts/03c_finalize_dataset.py` | Merge/dedup/split → `data/train.json`, `data/eval.json`. |
| `scripts/04_train.py` | QLoRA fine-tune (Unsloth) → `out/lora/`, `out/gguf/`. |
| `scripts/05_build_rag.py` | Build Chroma index (`code_r148` + `evolution`). |
| `scripts/06_rag_server.py` | OpenAI-compatible RAG proxy. |
| `scripts/07_eval.py` | Smoke eval against the proxy. |
| `serve/Modelfile` | Register the GGUF with Ollama. |
| `serve/continue_config.example.json` | VS Code Continue config. |

---

## Part 0 — Environment (WSL2 Ubuntu)

```bash
sudo apt update && sudo apt install -y python3.10-venv git git-lfs build-essential
python3 -m venv ~/cra && source ~/cra/bin/activate
pip install -r requirements.txt

# Training stack (separate step — Unsloth pulls compatible torch/bitsandbytes)
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
pip install trl peft accelerate datasets

python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
```

Run scripts from the `scripts/` dir (so `common.py` imports resolve), e.g.
`cd scripts && python 02_extract_commits.py`.

## Part 1 — Repo + history
Edit `config.yaml` → `repo.path` / `repo.url`, then:
```bash
bash scripts/01_setup_repo.sh
# copy the real 136.* / 148.* tags it prints into config.yaml -> versions.from/to
```

## Part 2 — Extract the evolution signal
```bash
cd scripts && python 02_extract_commits.py     # -> data/commits.jsonl
```

## Part 3 — Build the SFT dataset (hybrid)
```bash
cd scripts
python 03_make_sft_programmatic.py             # free, mechanical
export OPENAI_API_KEY=sk-...                    # teacher (set teacher_base_url in config for Claude)
python 03b_make_sft_teacher.py                 # distilled reasoning/"why" pairs
python 03c_finalize_dataset.py                 # -> data/train.json, data/eval.json
```
Aim for **5k–30k clean examples** for a first run. Quality > quantity.

## Part 4 — Fine-tune (QLoRA, ~11–13GB VRAM)
```bash
cd scripts && python 04_train.py               # -> out/lora/ (adapters), out/gguf/ (q4_k_m)
```
OOM? Lower `train.batch_size` to 1, `max_seq_len` to 1024, or `lora_r` to 16 in config.
Watch eval loss — 1–3 epochs is typical; rising eval loss = overfitting.

## Part 5 — Build RAG
```bash
cd scripts && python 05_build_rag.py           # -> rag_db/  (code_r148 + evolution)
```
Re-run after pulling new code — **no retraining needed**. This is what keeps the
assistant current.

## Part 6 — Serve + connect VS Code
```bash
# A) model endpoint — Ollama (simplest):
ollama create chromium-coder -f serve/Modelfile
ollama serve            # http://localhost:11434

#    ...or vLLM (faster, serves LoRA directly):
# vllm serve unsloth/Qwen2.5-Coder-7B-Instruct \
#   --enable-lora --lora-modules chromium=out/lora \
#   --max-model-len 8192 --quantization bitsandbytes
#   (then set serve.model_url to http://localhost:8000/v1/chat/completions)

# B) RAG proxy:
cd scripts && uvicorn 06_rag_server:app --port 9000

# C) VS Code: install the "Continue" extension, then merge
#    serve/continue_config.example.json into ~/.continue/config.json
```
Ask in the Continue panel: *"Why did RenderFrameHost lifetime handling change since 136?"*

## Part 7 — Evaluate
```bash
cp data/eval_questions.example.json data/eval_questions.json   # edit with real Q/A
cd scripts && python 07_eval.py
```
Compare base / base+RAG / FT+RAG. Expect **base+RAG** to win on recall and
**FT+RAG** to win on Chromium-idiomatic answers — that tells you the FT earns its keep.

---

## Maintenance cadence
- **Pulled new R148 code?** → re-run Part 5 only (minutes).
- **Conventions drifted / better data?** → re-run Parts 2–4 (hours). LoRA adapter is ~100MB.
