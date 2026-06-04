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
| `scripts/03_make_sft_programmatic.py` | Free "why" pairs (diff → rationale). |
| `scripts/03b_make_sft_teacher.py` | Teacher "why"/architecture pairs (needs API key). |
| `scripts/03d_make_sft_tasks_programmatic.py` | Free generation pairs (task → diff). |
| `scripts/03e_make_sft_tasks_teacher.py` | Teacher generation pairs (task → code + explanation). |
| `scripts/03c_finalize_dataset.py` | Merge/dedup/split all 4 → `data/train.json`, `data/eval.json`. |
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
The vibe repo upgrades annually via **milestone branches**: `origin/R148` is this
year's default, `origin/R136` was last year's (not Chromium tags). `config.yaml`
is preset to `versions.from_ref=origin/R136`, `to_ref=origin/R148`. Verify:
```bash
bash scripts/01_setup_repo.sh     # fetches, lists origin/R<NN> branches, checks anchors resolve & prints range count
```

## Part 2 — Extract the evolution signal (vibe + official)
```bash
cd scripts && python 02_extract_commits.py     # -> data/commits.jsonl
```
The `from_ref..to_ref` range is ~170k commits. The script **classifies provenance**
and selects what matters:
- **vibe** (downstream): GitHub-PR authors + `github-actions[bot]` R136→R148
  *porting* commits. **Kept in full, tree-wide** (~150) — the scarce company signal,
  incl. dirs outside `include_dirs` like `ash/`, `login/`.
- **official** (upstream Chromium): detected by `Cr-Commit-Position` /
  `chromium-review.googlesource.com` trailers or `@chromium.org`/`@google.com`,
  restricted to `include_dirs`.
- **noise**: autoroll / bisection / swarming service accounts + DEPS/V8 rolls — dropped.

**Two outputs (RAG coverage is decoupled from the fine-tune cap):**
- `data/commits_all.jsonl` — **every** de-noised commit (all vibe + all ~65k official),
  message-only. Feeds the **evolution RAG** index → full historical "why" coverage.
- `data/commits.jsonl` — all vibe + **sampled** official (`max_official_commits`), **with diffs**.
  Feeds the **SFT dataset** (Part 3). The cap keeps upstream commits from drowning the
  ~150 vibe ones; it does **not** limit what RAG can retrieve.

Each commit keeps an `origin` field that flows into the dataset and RAG metadata, so
the assistant can distinguish *"vibe customized this"* from *"upstream changed this."*

## Part 3 — Build the SFT dataset (hybrid)
The dataset has **two skills**: *understanding* ("why did this change") and
*generation* ("develop/refactor this"). Each has a free programmatic generator and
a teacher-distilled one.
```bash
cd scripts
export OPENAI_API_KEY=sk-...                    # teacher key (DeepSeek by default; see config)

# understanding ("why")
python 03_make_sft_programmatic.py             # free: diff -> rationale
python 03b_make_sft_teacher.py                 # distilled: why/architecture Q&A

# generation (develop / refactor / fix)
python 03d_make_sft_tasks_programmatic.py      # free: task -> unified diff
python 03e_make_sft_tasks_teacher.py           # distilled: task -> code + explanation

python 03c_finalize_dataset.py                 # merge all 4 -> data/train.json, data/eval.json
```
Aim for **5k–30k clean examples** for a first run. Quality > quantity. The task
generators (03d/03e) are what teach the model to **write** Chromium code, not just
explain it — without them the fine-tune only adapts *style*, and generation quality
falls back to base-model + RAG.

`03c` applies **per-source caps** (`dataset.source_caps`) so the bulky free
generators don't drown the scarce teacher pairs, and prints the resulting mix:
```
Balance: why=11500 (60%)  generation=7400 (39%)  | teacher=6900 (36%)
```
Tune the caps if the split looks off (e.g. lower `sft_programmatic.jsonl` toward
~4500 for a 50/50 why-vs-generation mix). Capping is seeded, so re-runs are stable.

## Part 4 — Fine-tune (QLoRA, ~11–13GB VRAM)
```bash
cd scripts && python 04_train.py               # -> out/lora/ (adapters), out/gguf/ (q4_k_m)
```
OOM? Lower `train.batch_size` to 1, `max_seq_len` to 1024, or `lora_r` to 16 in config.
Watch eval loss — 1–3 epochs is typical; rising eval loss = overfitting.

## Part 5 — Build RAG
```bash
cd scripts && python 05_build_rag.py                    # full build: code + evolution
# incremental & targeted modes:
python 05_build_rag.py --changed-since HEAD@{1}         # only re-embed files changed by last pull/rebase (fast)
python 05_build_rag.py --evolution                      # rebuild only the 'evolution' collection (after Part 2)
python 05_build_rag.py --code                           # rebuild only the 'code' collection
```
RAG holds the knowledge, so **re-indexing keeps the assistant current without
retraining**. Chunks/commits carry `origin` (`code`/`vibe`/`official`).

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

## Part 7 — Evaluate (A/B: is the fine-tune worth it?)
```bash
cp data/eval_questions.example.json data/eval_questions.json   # edit with real Q/A
cd scripts && python 07_eval.py
```
Retrieves the **same RAG context** per question and runs it through every reachable
system in `config.eval.systems` — `base+RAG`, `ft+RAG`, `frontier+RAG` — so you
compare *model quality at constant retrieval*. Prints a hit-rate/latency table and
writes `data/eval_results.json` with every answer.

**Run this BEFORE committing to fine-tuning.** It directly answers "do I need the
fine-tune?": if `base+RAG` already satisfies you, the FT's ROI is low. Unreachable
systems are auto-skipped, so you can start with just `base+RAG` + `frontier+RAG`
(no trained model yet) to see the ceiling, then add `ft+RAG` after Part 4.

Caveats: substring hit-rate undersells quality — **read the answers** in
`eval_results.json`. And `frontier+RAG` is a strong model with single-shot RAG, not
a tool-using agent; a real agent with live repo access could exceed it.

---

## When to run what (maintenance cadence)

The golden rule: **RAG is cheap and frequent; fine-tuning is expensive and rare.**
Knowledge lives in RAG, so refresh it often. Only re-fine-tune when *behavior/style*
needs to change — batch it.

| Trigger | Run | Cost | Why |
|---|---|---|---|
| **Every `git pull` / rebase** on your working branch (R148 code moved) | `python 05_build_rag.py --changed-since HEAD@{1}` | seconds–minutes | Keep the `code` index matching the tree you actually edit. Incremental = only changed files. |
| **Weekly / occasional full code refresh** (drift, new dirs added to `include_dirs`) | `python 05_build_rag.py --code` | minutes | Catch additions/deletions a `--changed-since` chain may have missed. |
| **New vibe CLs merged** or the milestone branch advances (new upstream pulled in) | `python 02_extract_commits.py` → `python 05_build_rag.py --evolution` | minutes | Refresh the "why" (commits + rationale). No retraining — it's just the evolution index. |
| **Enough *new* commits to be worth distilling** (e.g. a sprint's worth of vibe CLs) | Parts 2 → 3 → 3c, then **append** to your dataset | API \$ + minutes | Grow the SFT set. Don't retrain yet — accumulate. |
| **≥3 milestone migrations accumulated**, OR house conventions clearly drifted, OR the dataset grew substantially since last train | Parts 4 (re-fine-tune) → re-register GGUF (Part 6A) | hours (1 GPU) | Refresh *behavior/idioms* in the weights. LoRA adapter is ~100MB; cheap to swap. |
| **Annual upgrade** (next year: R148 → R160 becomes default) | Set `versions.from_ref=origin/R148`, `to_ref=origin/R160`; rerun Parts 1→5; consider a fresh fine-tune | half a day | New baseline + new evolution window. |

Rules of thumb:
- **Never** re-fine-tune just because code changed — that's what RAG is for.
- After Part 2 you must rebuild the **evolution** index (`--evolution`) for new commits to be retrievable.
- After Part 4 you must re-create the Ollama model / reload the vLLM adapter for new weights to take effect.
- Keep `data/*.jsonl` so dataset growth is append-only across milestones (more diverse data = better fine-tune).
