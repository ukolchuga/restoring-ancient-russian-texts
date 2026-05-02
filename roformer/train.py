#!/usr/bin/env python3
"""
Main training script for the Ancient Russian RoFormer model.
Features custom MLM collator, Test A (on-the-fly evaluation), and Test B (CER/EM restoration evaluation).
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import evaluate as hf_evaluate
import numpy as np
import pandas as pd
import torch
from collator import RoFormerPhysicalDegradationCollator
from datasets import load_from_disk
from model import get_model
from tqdm import tqdm
from transformers import RobertaTokenizerFast, Trainer, TrainingArguments

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def save_json(data: dict, path: str | Path) -> None:
    """Save data to a formatted JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"Saved: {path}")


def preprocess_logits_for_metrics(logits, labels):
    """Efficiently extract top predictions for metric calculation during training (Test A)."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.topk(logits, k=5, dim=-1).indices


def compute_metrics(eval_preds):
    """Calculate Top-1, Top-3, and Top-5 accuracy for masked positions (Test A)."""
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


def evaluate_test_b_final(
    model, tokenizer, test_b_dataset, output_path: Path, device: torch.device
):
    """
    Test B: Dynamic Evaluation of Lacunae Restoration.
    Uses offset_mapping to mask specific BPE tokens corresponding to physical damage.
    Calculates Exact Match (EM) and Character Error Rate (CER).
    """
    log.info("Running Final Test B (Lacunae Restoration)...")
    cer_metric = hf_evaluate.load("cer")

    exact_matches = 0
    total_samples = 0
    predictions_list = []
    references_list = []
    report_rows = []

    model.eval()

    for i, record in enumerate(tqdm(test_b_dataset, desc="Evaluating Test B")):
        orig_text = record["original"]
        target_text = record["target_text"]

        # 1. Tokenize target text with offset mapping
        encoded = tokenizer(
            target_text, return_offsets_mapping=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"][0].clone()
        offsets = encoded["offset_mapping"][0]

        # 2. Find spans (character coordinates) of lacunae in target text
        spans_to_mask = []
        orig_idx = 0
        pattern = re.compile(
            r"\(([^)]+)\)|\[(?!(?:GAP|MASK|PAD|UNK|CLS|SEP)\]|CTX_)([^\]]+)\]"
        )

        for match in pattern.finditer(orig_text):
            span_text = match.group(1) or match.group(2)
            start_idx = target_text.find(span_text, orig_idx)
            if start_idx != -1:
                end_idx = start_idx + len(span_text)
                spans_to_mask.append((start_idx, end_idx))
                orig_idx = end_idx

        if not spans_to_mask:
            continue

        # 3. Identify BPE tokens intersecting with lacunae spans
        mask_token_indices = []
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:  # Skip special tokens
                continue
            for span_start, span_end in spans_to_mask:
                if max(tok_start, span_start) < min(tok_end, span_end):
                    mask_token_indices.append(idx)
                    break

        if not mask_token_indices:
            continue

        # 4. Apply masks
        masked_input_ids = input_ids.clone()
        for idx in mask_token_indices:
            masked_input_ids[idx] = tokenizer.mask_token_id

        # 5. Model Inference
        masked_input_ids = masked_input_ids.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(masked_input_ids)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu()

        # 6. Reconstruct text with predicted tokens
        final_ids = input_ids.clone()
        for idx in mask_token_indices:
            final_ids[idx] = predictions[idx]

        predicted_text = tokenizer.decode(final_ids, skip_special_tokens=True)
        # Target text also decoded to ensure spacing aligns perfectly with prediction
        clean_target = tokenizer.decode(input_ids, skip_special_tokens=True)

        predictions_list.append(predicted_text)
        references_list.append(clean_target)

        is_exact = predicted_text == clean_target
        if is_exact:
            exact_matches += 1
        total_samples += 1

        # Save to report
        report_rows.append(
            {
                "sample_idx": i,
                "original_annotated": orig_text,
                "target": clean_target,
                "predicted": predicted_text,
                "exact_match": is_exact,
            }
        )

    # 7. Compute Metrics
    metrics = {"total_samples": total_samples, "exact_match_pct": 0.0, "cer_pct": 0.0}
    if total_samples > 0:
        metrics["exact_match_pct"] = round((exact_matches / total_samples) * 100, 2)
        metrics["cer_pct"] = round(
            cer_metric.compute(predictions=predictions_list, references=references_list)
            * 100,
            2,
        )

        log.info(f"✅ Test B - Total Evaluated: {total_samples}")
        log.info(f"✅ Test B - Exact Match: {metrics['exact_match_pct']}%")
        log.info(f"✅ Test B - CER: {metrics['cer_pct']}%")

        # Save detailed report
        pd.DataFrame(report_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        log.info(f"Detailed Test B report saved to {output_path}")
    else:
        log.warning("❌ No valid lacunae found for Test B evaluation.")

    return metrics


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
    parser.add_argument("--report_test_b", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Starting RoFormer Training Pipeline on {device}")

    # Load dataset and tokenizer
    dataset = load_from_disk(args.dataset_dir)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_path)

    special_tokens_dict = {
        "additional_special_tokens": [
            "[CTX_CHURCH]",
            "[CTX_DAILY]",
            "[CTX_LEGAL]",
            "[CTX_LIT]",
            "[CTX_EPIC]",
            "[CTX_SCIENCE]",
            "[GAP]",
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    log.info(f"Training blocks: {len(dataset['train']):,}")
    log.info(f"Test A blocks: {len(dataset['test_a']):,}")
    log.info(f"Vocab size: {len(tokenizer):,}")

    # Initialize model
    model = get_model(len(tokenizer), tokenizer.pad_token_id)
    model.resize_token_embeddings(len(tokenizer))

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
        logging_steps=50,
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

    trainer = LoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test_a"],  # Test A evaluated continually
        data_collator=RoFormerPhysicalDegradationCollator(
            tokenizer=tokenizer, mlm_prob=0.15, max_span=3, edge_prob=0.15
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        log_path=log_path,
    )

    # 1. Start training
    trainer.train()

    # 2. Final Evaluation of Test A
    log.info("Computing final metrics for Test A (Perplexity & Accuracy)...")
    trainer.data_collator.add_random_gaps = False  # Cleaner evaluation
    eval_metrics = trainer.evaluate()
    # Add Perplexity calculation
    eval_metrics["eval_perplexity"] = round(np.exp(eval_metrics["eval_loss"]), 4)
    save_json(eval_metrics, output_dir / f"eval_metrics_test_a_{timestamp}.json")
    log.info(f"Final Test A Perplexity: {eval_metrics['eval_perplexity']}")

    # 3. Final Evaluation of Test B (Business Metrics)
    if args.report_test_b and "test_b" in dataset:
        test_b_metrics = evaluate_test_b_final(
            model=model,
            tokenizer=tokenizer,
            test_b_dataset=dataset["test_b"],
            output_path=output_dir / f"test_b_predictions_{timestamp}.csv",
            device=device,
        )
        save_json(test_b_metrics, output_dir / f"metrics_test_b_{timestamp}.json")

    # 4. Save final artifacts
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    log.info("🎉 Training complete. Models, logs, and reports saved.")


if __name__ == "__main__":
    main()
