"""Part 4 — QLoRA fine-tune Qwen2.5-Coder-7B with Unsloth (run inside WSL2).

Requires the training stack (see README Part 0):
  pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
  pip install trl peft accelerate datasets

Outputs:
  out/lora/        LoRA adapters (~100MB) — serve with vLLM
  out/gguf/        merged q4_k_m GGUF     — serve with Ollama/llama.cpp
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from common import load_config, data_path, root_path

cfg = load_config()
t = cfg["train"]
MAXLEN = t["max_seq_len"]

model, tok = FastLanguageModel.from_pretrained(
    t["base_model"], max_seq_length=MAXLEN, load_in_4bit=True, dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model, r=t["lora_r"], lora_alpha=t["lora_alpha"], lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=0,
)

ds = load_dataset("json", data_files={
    "train": data_path("train.json"),
    "eval": data_path("eval.json"),
})


def fmt(ex):
    return {"text": tok.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}


ds = ds.map(fmt)

trainer = SFTTrainer(
    model=model, tokenizer=tok,
    train_dataset=ds["train"], eval_dataset=ds["eval"],
    args=SFTConfig(
        dataset_text_field="text", max_seq_length=MAXLEN,
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        warmup_ratio=0.03, num_train_epochs=t["epochs"],
        learning_rate=t["learning_rate"], bf16=True, logging_steps=10,
        optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="cosine",
        output_dir=root_path(t["output_dir"]),
        save_steps=200, eval_strategy="steps", eval_steps=200,
    ),
)
# Compute loss only on assistant turns (Qwen2.5 chat markers).
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

trainer.train()

lora_dir = root_path(t["output_dir"], "lora")
model.save_pretrained(lora_dir)
tok.save_pretrained(lora_dir)
print(f"Saved LoRA adapters -> {lora_dir}")

gguf_dir = root_path(t["output_dir"], "gguf")
model.save_pretrained_gguf(gguf_dir, tok, quantization_method=t["gguf_quant"])
print(f"Saved GGUF -> {gguf_dir}")
