#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase-1 SFT for Mistral-7B-Instruct-v0.3 using DeepSpeed.

- Reads SFT items from a JSONL file with fields: {prompt, target, ...}.
- Builds sequences: [BOS] + prompt + '\n' + target + [EOS]
- Masks labels on the prompt (and padding), trains only on target tokens.
- Keeps the full target by truncating the prompt from the LEFT if needed.
- Uses Transformers Trainer with DeepSpeed.

Usage (single node, 2 GPUs example):
  deepspeed --num_gpus=2 train_sft_mistral.py \
    --model_id "/reference/LLMs/Mistral_AI/mistral-7B-Instruct-v0.3-hf/" \
    --train_file "./out_sft/sft_items.jsonl" \
    --output_dir "./results/mistral7b" \
    --deepspeed "./deepspeed_config.json" \
    --epochs 2 --per_device_train_batch_size 1 --gradient_accumulation_steps 16

Or with torchrun (DeepSpeed still used via Trainer arg):
  torchrun --nproc_per_node=2 train_sft_mistral.py  ...same args...
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import deepspeed

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

IGNORE_INDEX = -100

print("Cuda support:", torch.cuda.is_available(),":", torch.cuda.device_count(), "devices")
deepspeed.init_distributed(dist_backend="nccl")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Devices visible to PyTorch: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")



# ----------------------------
# Data: JSONL reader + split
# ----------------------------

def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}")
            if "prompt" not in obj or "target" not in obj:
                raise ValueError(f"Line {i} missing 'prompt' or 'target' field.")
            prompt = str(obj["prompt"]).strip()
            target = str(obj["target"]).strip()
            if prompt == "" or target == "":
                # skip empty items
                continue
            items.append({"prompt": prompt, "target": target, "meta": obj.get("meta", {}), "id": obj.get("id", f"item{i:06d}")})
    if not items:
        raise ValueError("No valid items found in JSONL.")
    return items


def split_train_eval(
    items: List[Dict],
    eval_ratio: float = 0.05,
    seed: int = 42
) -> (List[Dict], List[Dict]):
    rng = random.Random(seed)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    n_eval = max(1, int(len(items) * eval_ratio))
    eval_ids = set(idxs[:n_eval])
    train, eval_ = [], []
    for i, ex in enumerate(items):
        (eval_ if i in eval_ids else train).append(ex)
    return train, eval_


# ----------------------------
# Dataset and Collator
# ----------------------------

class SFTJsonlDataset(Dataset):
    """Returns raw dicts {prompt, target, id, meta}."""
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]


@dataclass
class DataCollatorPromptTarget:
    """
    Tokenizes batch of {prompt, target} into:
      input_ids, attention_mask, labels
    Sequence = [BOS?] + prompt + "\n" + target + [EOS]
    - labels = -100 for prompt & padding; real ids for target (incl. EOS).
    - Left-truncate prompt ONLY so target is preserved.
    """
    tokenizer: AutoTokenizer
    max_length: int = 2048
    add_bos: bool = True
    add_eos: bool = True
    keep_separator_newline: bool = True  # insert '\n' between prompt and target
    truncate_target: bool = True         # if target itself exceeds budget, truncate tail

    def __call__(self, batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []
        attn_batch: List[List[int]] = []

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is None and eos_id is not None:
            # Ensure pad token exists (use eos)
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for ex in batch:
            prompt_text = ex["prompt"].strip()
            target_text = ex["target"].strip()

            # Optional separator newline to make the boundary explicit
            sep = "\n" if self.keep_separator_newline else " "

            # Tokenize without special tokens so counts align
            prompt_ids = self.tokenizer.encode(
                prompt_text + sep, add_special_tokens=False
            )
            target_ids = self.tokenizer.encode(
                target_text, add_special_tokens=False
            )

            # Append EOS to target if requested and not already present
            if self.add_eos and (eos_id is not None):
                if len(target_ids) == 0 or target_ids[-1] != eos_id:
                    target_ids = target_ids + [eos_id]

            # Budget accounting: reserve for BOS (optional) + prompt + target
            reserved_for_bos = 1 if (self.add_bos and bos_id is not None) else 0
            total_len = reserved_for_bos + len(prompt_ids) + len(target_ids)

            # If too long, first try to keep full target and trim prompt from LEFT
            if total_len > self.max_length:
                # Room available for prompt after accounting for BOS & target
                room_for_prompt = self.max_length - reserved_for_bos - len(target_ids)
                if room_for_prompt < 0:
                    # Even target doesn't fit fully. Optionally truncate target tail.
                    if self.truncate_target:
                        # Keep the last part of target (right-most) and drop its head
                        keep_tgt = max(1, self.max_length - reserved_for_bos)  # no prompt
                        target_ids = target_ids[-keep_tgt:]
                        room_for_prompt = 0
                    else:
                        # Drop this example if you prefer strictness (rare)
                        target_ids = target_ids[-(self.max_length - reserved_for_bos):]
                        room_for_prompt = 0

                # Now trim prompt from the LEFT so the prompt length <= room_for_prompt
                if len(prompt_ids) > max(0, room_for_prompt):
                    prompt_ids = prompt_ids[-max(0, room_for_prompt):]

            # Build final sequence
            seq_ids: List[int] = []
            if self.add_bos and bos_id is not None:
                seq_ids.append(bos_id)
            seq_ids.extend(prompt_ids)
            seq_ids.extend(target_ids)

            # Build labels: -100 for BOS+prompt, target labels equal to ids
            n_prompt = (1 if (self.add_bos and bos_id is not None) else 0) + len(prompt_ids)
            labels = [IGNORE_INDEX] * n_prompt + target_ids[:]

            attn = [1] * len(seq_ids)

            input_ids_batch.append(seq_ids)
            labels_batch.append(labels)
            attn_batch.append(attn)

        # Pad to max in batch
        maxlen = max(len(x) for x in input_ids_batch)
        pad_id = self.tokenizer.pad_token_id

        def pad_to(x: List[int], pad_val: int) -> List[int]:
            return x + [pad_val] * (maxlen - len(x))

        input_ids = torch.tensor([pad_to(x, pad_id) for x in input_ids_batch], dtype=torch.long)
        labels = torch.tensor([pad_to(x, IGNORE_INDEX) for x in labels_batch], dtype=torch.long)
        attention_mask = torch.tensor([pad_to(x, 0) for x in attn_batch], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# ----------------------------
# Main: Trainer + DeepSpeed
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str,default="/reference/LLMs/Mistral_AI/mistral-7B-Instruct-v0.3-hf/",
                        help="Local path or HF id for Mistral-7B-Instruct-v0.3")
    parser.add_argument("--train_file", type=str, default="out_sft/sft_items.jsonl",
                        help="Path to sft_items.jsonl")
    parser.add_argument("--output_dir", type=str, default='results/mistral_sft_full',
                        help="Where to save checkpoints and final model")
    parser.add_argument("--deepspeed", type=str,default='./deepspeed_config.json',
                        help="Path to DeepSpeed config JSON")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--report_to", type=str, default="none",
                        choices=["none", "wandb", "tensorboard"])
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--local_rank", type=int,
                    default=int(os.environ.get("LOCAL_RANK", -1)),
                    help="Provided by DeepSpeed/torchrun.")

    args, _ = parser.parse_known_args()


    set_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---- Load data
    all_items = read_jsonl(args.train_file)
    train_items, eval_items = split_train_eval(all_items, eval_ratio=args.eval_ratio, seed=args.seed)
    print(f"[Data] Train: {len(train_items)} | Eval: {len(eval_items)} | Total: {len(all_items)}")

    train_ds = SFTJsonlDataset(train_items)
    eval_ds = SFTJsonlDataset(eval_items)

    # ---- Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, use_fast=True, local_files_only=args.local_files_only
    )
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    # Gradient checkpointing + no cache
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if getattr(model.config, "use_cache", None):
        model.config.use_cache = False

    # ---- Collator
    collator = DataCollatorPromptTarget(
        tokenizer=tokenizer,
        max_length=args.max_length,
        add_bos=True,
        add_eos=True,
        keep_separator_newline=True,
        truncate_target=True,
    ) 

    # ---- Training args (DeepSpeed integrated)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        deepspeed=args.deepspeed,
        report_to=None if args.report_to == "none" else args.report_to,
        # dataloader_num_workers=4,
        # group_by_length=True,                   # better packing by length
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=1.0,
        logging_first_step=True,
        remove_unused_columns=False,            # VERY IMPORTANT for causal LM + custom collator
        torch_compile=False,                    # leave False with DeepSpeed ZeRO
        label_smoothing_factor=args.label_smoothing,
        ddp_find_unused_parameters=False,
    )

    # ---- Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # ---- Train
    trainer.train()

    # ---- Save final
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ---- Eval perplexity (quick)
    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss", None)
    if eval_loss is not None:
        try:
            ppl = math.exp(eval_loss)
        except OverflowError:
            ppl = float("inf")
        print(f"[Eval] loss={eval_loss:.4f} | ppl={ppl:.2f}")
        metrics["perplexity"] = ppl
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_state()

    print("Done.")


if __name__ == "__main__":
    main()
