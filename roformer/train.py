#!/usr/bin/env python3
"""
Main training script for the Ancient Russian RoFormer model.
Features custom MLM collator, stratified evaluation, and detailed prediction reporting.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from collator import RoFormerPhysicalDegradationCollator
from datasets import load_from_disk
from model import get_model
from transformers import RobertaTokenizerFast, Trainer, TrainingArguments

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def load_json(path: str | Path) -> dict:
    """Load data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str | Path) -> None:
    """Save data to a formatted JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"Saved: {path}")


def preprocess_logits_for_metrics(logits, labels):
    """Efficiently extract top predictions for metric calculation during training."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.topk(logits, k=5, dim=-1).indices


def compute_metrics(eval_preds):
    """Calculate Top-1, Top-3, and Top-5 accuracy for masked positions."""
    preds, labels = eval_preds
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    if labels.size == 0:
        return {"top1_accuracy": 0.0, "top3_accuracy": 0.0, "top5_accuracy": 0.0}

    return {
        "top1_accuracy": float(np.mean(preds[:, 0] == labels)),
        "top3_accuracy": float(
            np.mean(np.any(preds[:, :3] == labels[:, None], axis=1))
        ),
        "top5_accuracy": float(
            np.mean(np.any(preds[:, :5] == labels[:, None], axis=1))
        ),
    }


class LoggingTrainer(Trainer):
    """Custom Trainer subclass that saves training logs to a JSON file incrementally."""

    def __init__(self, *args, log_path: Optional[Path] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path
        self.log_history = []

    def log(self, logs):
        super().log(logs)
        self.log_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                **logs,
            }
        )
        if self.log_path and len(self.log_history) % 10 == 0:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.log_history, f, ensure_ascii=False, indent=2)


def decode_one_token(tokenizer, token_id: int) -> str:
    """Decode a single token ID into text, handling BPE-specific space artifacts."""
    if token_id < 0 or token_id >= len(tokenizer):
        return "[UNK]"
    token = tokenizer.convert_ids_to_tokens([token_id])[0]
    if token.startswith("Ġ"):
        token = " " + token[1:]
    return token


def generate_predictions_report(
    model,
    tokenizer,
    dataset,
    output_path: Path,
    max_samples: int = 100,
    k_values: tuple = (1, 3, 5),
    device: Optional[torch.device] = None,
) -> dict:
    """
    Generate a detailed CSV report comparing model predictions against ground truth labels.
    Calculates hit@k metrics specifically for the evaluation set.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    rows = []
    total_samples = min(len(dataset), max_samples)
    top_k_max = max(k_values)

    log.info(f"Generating detailed prediction report for {total_samples} samples...")

    hit_accum = {f"hit@{k}": 0 for k in k_values}
    correct = 0
    used = 0

    for i, sample in enumerate(dataset.select(range(total_samples))):
        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            [sample["attention_mask"]], dtype=torch.long, device=device
        )
        labels = sample.get("labels", None)

        if labels is None or -100 not in labels:
            continue

        mask_positions = [j for j, l in enumerate(labels) if l != -100]
        if not mask_positions:
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]

        for pos in mask_positions:
            true_id = labels[pos]
            if true_id < 0:
                continue

            pred_logits = logits[pos]
            top_ids = (
                torch.topk(pred_logits, k=min(top_k_max, len(tokenizer)))
                .indices.cpu()
                .numpy()
            )

            pred_id = int(top_ids[0])
            true_token = decode_one_token(tokenizer, true_id)
            pred_token = decode_one_token(tokenizer, pred_id)

            if pred_token.strip().startswith("[") or true_token.strip().startswith("["):
                continue

            used += 1
            is_correct = pred_id == true_id
            if is_correct:
                correct += 1

            probs = torch.softmax(pred_logits, dim=-1)
            top1_prob = float(probs[pred_id].item())

            true_rank = next(
                (rank + 1 for rank, tid in enumerate(top_ids) if tid == true_id), None
            )
            top_tokens = [decode_one_token(tokenizer, int(tid)) for tid in top_ids[:5]]

            for k in k_values:
                if any(int(tid) == true_id for tid in top_ids[:k]):
                    hit_accum[f"hit@{k}"] += 1

            rows.append(
                {
                    "sample_idx": i,
                    "position": pos,
                    "true_token": true_token,
                    "pred_token": pred_token,
                    "is_correct": is_correct,
                    "true_rank": true_rank,
                    "top1_prob": round(top1_prob, 4),
                    "top5_preds": "|".join(top_tokens),
                }
            )

    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        log.info(f"Prediction report saved: {output_path}")

    return {
        "total_predictions": used,
        "correct": correct,
        "accuracy": round(correct / used, 4) if used > 0 else 0.0,
        **{k: round(v / used, 4) if used > 0 else 0.0 for k, v in hit_accum.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="RoFormer Trainer for Ancient Russian."
    )
    parser.add_argument("--dataset_dir", default="artifacts/roformer_dataset")
    parser.add_argument("--tokenizer_path", default="ancient_rus_tokenizer_BPE")
    parser.add_argument("--output_dir", default="artifacts/roformer_training_output")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train_bs", type=int, default=32)
    parser.add_argument("--eval_bs", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=400)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_test_b", action="store_true")
    parser.add_argument("--max_report_samples", type=int, default=1000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting RoFormer Training Pipeline")

    # Load dataset and tokenizer
    dataset = load_from_disk(args.dataset_dir)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_path)

    # Define and add all special tokens requested by the user
    special_tokens_dict = {
        "additional_special_tokens": [
            "[CTX_CHURCH]",
            "[CTX_DAILY]",
            "[CTX_LEGAL]",
            "[CTX_LIT]",
            "[CTX_EPIC]",
            "[CTX_SCIENCE]",
            "[GAP]",
            "·",
            ":",
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    log.info(f"Training samples: {len(dataset['train']):,}")
    log.info(f"Vocab size: {len(tokenizer):,}")

    # Initialize model
    model = get_model(len(tokenizer), tokenizer.pad_token_id)
    model.resize_token_embeddings(len(tokenizer))

    # Configure training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"training_log_{timestamp}.json"

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=3,
        logging_steps=100,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        fp16=args.fp16,
        dataloader_num_workers=4,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="top1_accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        seed=args.seed,
    )

    # Initialize Trainer
    trainer = LoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test_a"],
        data_collator=RoFormerPhysicalDegradationCollator(
            tokenizer=tokenizer, mlm_prob=0.12, max_span=3, edge_prob=0.15
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        log_path=log_path,
    )

    # Start training
    trainer.train()

    # Final evaluation and reporting
    log.info("Computing final metrics...")
    trainer.data_collator.add_random_gaps = False  # Disable augmentation for final eval
    eval_metrics = trainer.evaluate()
    save_json(eval_metrics, output_dir / f"eval_metrics_{timestamp}.json")

    if args.report_test_b and "test_b" in dataset:
        report_metrics = generate_predictions_report(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset["test_b"],
            output_path=output_dir / f"predictions_report_{timestamp}.csv",
            max_samples=args.max_report_samples,
        )
        save_json(
            report_metrics, output_dir / f"predictions_report_metrics_{timestamp}.json"
        )

    # Save final artifacts
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    log.info("Training complete. Models and logs saved.")


if __name__ == "__main__":
    main()
