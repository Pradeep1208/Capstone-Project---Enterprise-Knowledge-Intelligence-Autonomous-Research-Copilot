#!/usr/bin/env python
"""QLoRA fine-tuning (optional work package). Needs a CUDA GPU, e.g. a free Colab T4. NOT used by the Streamlit app.

  pip install -r finetune/requirements-finetune.txt
  python finetune/prepare_hotpot.py
  python finetune/lora_train.py --model Qwen/Qwen2.5-1.5B-Instruct

Check the base model's licence before use. Compare base vs adapted model on the SAME benchmark prompts.
"""
from __future__ import annotations

import argparse
import json

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)


def encode(example: dict, tok, max_len: int) -> dict:
    prompt = tok.apply_chat_template([{"role": "user", "content": example["prompt"]}], tokenize=False,
                                     add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(prompt + example["response"] + tok.eos_token, add_special_tokens=False, truncation=True,
              max_length=max_len)["input_ids"]
    labels = ([-100] * len(p_ids) + ids[len(p_ids):])[: len(ids)]       # train on the response only
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--data", default="finetune/outputs/data")
    p.add_argument("--out", default="finetune/outputs/adapter")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=1536)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                             if torch.cuda.is_bf16_supported() else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def load(name):
        with open(f"{a.data}/{name}", encoding="utf-8") as f:
            return [encode(json.loads(l), tok, a.max_len) for l in f if l.strip()]

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, per_device_train_batch_size=2,
        gradient_accumulation_steps=8, logging_steps=20, save_strategy="no", report_to="none",
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(), lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )
    Trainer(model=model, args=args, train_dataset=load("train.jsonl"),
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"Adapter saved to {a.out}")


if __name__ == "__main__":
    main()
