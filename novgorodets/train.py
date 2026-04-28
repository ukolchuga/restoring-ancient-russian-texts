#!/usr/bin/env python3
"""
Актуальный трейнер для DualBert с полным логированием и репортом предсказаний.

Сохраняет:
- Логи тренировки (JSON)
- Метрики валидации (CSV)
- Финальный репорт (JSON)
- Репорт предсказаний на тестовых данных (CSV)
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments

from config import DualBertConfig
from model import DualBertForMaskedLM
from collator import DualPhysicalDegradationCollator

# ============================================================================
# ЛОГИРОВАНИЕ
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


# ============================================================================
# УТИЛИТЫ
# ============================================================================

def load_json(path: str | Path) -> dict:
    """Загружает JSON файл."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str | Path) -> None:
    """Сохраняет данные в JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"Saved: {path}")


# ============================================================================
# МЕТРИКИ
# ============================================================================

def preprocess_logits_for_metrics(logits, labels):
    """Подготавливает logits для расчета top-k метрик."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.topk(logits, k=5, dim=-1).indices


def compute_metrics(eval_preds):
    """Расчет top-k точности."""
    preds, labels = eval_preds
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    if labels.size == 0:
        return {
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "top5_accuracy": 0.0,
        }

    return {
        "top1_accuracy": float(np.mean(preds[:, 0] == labels)),
        "top3_accuracy": float(np.mean(np.any(preds[:, :3] == labels[:, None], axis=1))),
        "top5_accuracy": float(np.mean(np.any(preds[:, :5] == labels[:, None], axis=1))),
    }


# ============================================================================
# ТРЕЙНЕР С ЛОГИРОВАНИЕМ
# ============================================================================

class LoggingTrainer(Trainer):
    """Трейнер с сохранением логов в JSON."""

    def __init__(self, *args, log_path: Optional[Path] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path
        self.log_history = []

    def log(self, logs):
        """Переопределяем логирование для сохранения в файл."""
        super().log(logs)

        # Сохраняем логи
        self.log_history.append({
            "timestamp": datetime.now().isoformat(),
            **logs,
        })

        # Периодически сохраняем на диск
        if self.log_path and len(self.log_history) % 10 == 0:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.log_history, f, ensure_ascii=False, indent=2)


# ============================================================================
# РЕПОРТ ПРЕДСКАЗАНИЙ
# ============================================================================

def decode_one_token(tokenizer, token_id: int) -> str:
    """Декодирует один токен в символ."""
    if token_id < 0 or token_id >= len(tokenizer):
        return "[UNK]"
    token = tokenizer.convert_ids_to_tokens([token_id])[0]
    # Убираем префиксы BPE/WordPiece если есть
    if token.startswith("##"):
        token = token[2:]
    return token


def generate_predictions_report(
    model,
    tokenizer,
    char_vocab: dict,
    word_vocab: dict,
    dataset,
    output_path: Path,
    max_samples: int = 100,
    k_values: tuple = (1, 3, 5),
    device: Optional[torch.device] = None,
) -> dict:
    """
    Генерирует детальный репорт предсказаний модели.

    Args:
        model: Обученная модель
        tokenizer: Токенизатор
        char_vocab: Словарь символов
        word_vocab: Словарь слов
        dataset: Датасет для оценки
        output_path: Путь для сохранения CSV
        max_samples: Максимум примеров для анализа
        k_values: Значения k для метрик hit@k
        device: Устройство (gpu/cpu)

    Returns:
        dict с метриками
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    rows = []

    total_samples = min(len(dataset), max_samples)
    top_k_max = max(k_values)

    log.info(f"Generating predictions report for {total_samples} samples...")

    hit_accum = {f"hit@{k}": 0 for k in k_values}
    correct = 0
    used = 0

    for i, sample in enumerate(dataset.select(range(total_samples))):
        if i % 20 == 0:
            log.info(f"  Processing {i}/{total_samples}...")

        # Подготовка данных
        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
        word_ids = torch.tensor([sample["word_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            [sample["attention_mask"]], dtype=torch.long, device=device
        )
        labels = sample.get("labels", None)

        if labels is None or -100 not in labels:
            continue

        # Позиции с маской
        mask_positions = [j for j, l in enumerate(labels) if l != -100]
        if not mask_positions:
            continue

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                word_ids=word_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Обработка каждой маскированной позиции
        for pos in mask_positions:
            true_id = labels[pos]
            if true_id < 0:
                continue

            pred_logits = logits[pos]
            top_ids = torch.topk(pred_logits, k=min(top_k_max, len(char_vocab))).indices
            top_ids = top_ids.cpu().numpy()

            # Декодируем топ предсказания
            pred_id = int(top_ids[0])
            true_char = decode_one_token(tokenizer, true_id)
            pred_char = decode_one_token(tokenizer, pred_id)

            # Пропускаем специальные токены в оценке
            if pred_char.startswith("[") or true_char.startswith("["):
                continue

            used += 1
            is_correct = (pred_id == true_id)
            if is_correct:
                correct += 1

            # Вероятность лучшего предсказания
            probs = torch.softmax(pred_logits, dim=-1)
            top1_prob = float(probs[pred_id].item())

            # Ранг истинного символа
            true_rank = None
            for rank, tid in enumerate(top_ids):
                if tid == true_id:
                    true_rank = rank + 1
                    break

            # Топ-5 предсказаний
            top_chars = [decode_one_token(tokenizer, int(tid)) for tid in top_ids[:5]]

            # Проверяем hit@k
            for k in k_values:
                if any(decode_one_token(tokenizer, int(tid)) == true_char
                       for tid in top_ids[:k]):
                    hit_accum[f"hit@{k}"] += 1

            rows.append({
                "sample_idx": i,
                "position": pos,
                "true_char": true_char,
                "pred_char": pred_char,
                "is_correct": is_correct,
                "true_rank": true_rank,
                "top1_prob": round(top1_prob, 4),
                "top5_preds": "|".join(top_chars),
            })

    # Сохраняем репорт
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        log.info(f"Saved predictions report: {output_path}")

    # Итоговые метрики
    metrics = {
        "total_predictions": used,
        "correct": correct,
        "accuracy": round(correct / used, 4) if used > 0 else 0.0,
        **{
            k: round(v / used, 4) if used > 0 else 0.0
            for k, v in hit_accum.items()
        },
    }

    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Трейнер DualBert с полным логированием"
    )

    # Пути
    parser.add_argument(
        "--dataset_dir",
        default="novgorodets/artifacts/dual_dataset",
        help="Директория с датасетом",
    )
    parser.add_argument(
        "--char_vocab_path",
        default="novgorodets/artifacts/char_tokenizer/char_vocab.json",
        help="Путь к словарю символов",
    )
    parser.add_argument(
        "--word_vocab_path",
        default="novgorodets/artifacts/word_vocab.json",
        help="Путь к словарю слов",
    )
    parser.add_argument(
        "--output_dir",
        default="novgorodets/artifacts/training_output",
        help="Директория для результатов тренировки",
    )

    # Параметры модели
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)

    # Параметры тренировки
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_bs", type=int, default=32)
    parser.add_argument("--eval_bs", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=400)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Репорт
    parser.add_argument(
        "--report_test_b",
        action="store_true",
        help="Репорт на test_b",
    )
    parser.add_argument(
        "--max_report_samples",
        type=int,
        default=1000,
        help="Максимум образцов для репорта",
    )

    args = parser.parse_args()

    # ========================================================================
    # ПОДГОТОВКА
    # ========================================================================

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 80)
    log.info("Starting DualBert Training")
    log.info("=" * 80)
    log.info(f"Output directory: {output_dir}")

    # Загружаем данные
    log.info("Loading dataset...")
    dataset = load_from_disk(args.dataset_dir)
    char_vocab = load_json(args.char_vocab_path)
    word_vocab = load_json(args.word_vocab_path)

    log.info(f"  Train samples: {len(dataset['train']):,}")
    log.info(f"  Eval samples:  {len(dataset['test_a']):,}")
    log.info(f"  Char vocab size: {len(char_vocab):,}")
    log.info(f"  Word vocab size: {len(word_vocab):,}")

    # ========================================================================
    # МОДЕЛЬ
    # ========================================================================

    log.info("Creating model...")
    config = DualBertConfig(
        vocab_char_size=len(char_vocab),
        vocab_word_size=len(word_vocab),
        word_char_emb_dim=192,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.max_len,
        pad_token_id=char_vocab["[PAD]"],
    )
    model = DualBertForMaskedLM(config)

    # Подсчитываем параметры
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Total params: {n_params:,}")
    log.info(f"  Trainable params: {n_trainable:,}")

    # ========================================================================
    # КОЛЛАТОР
    # ========================================================================

    special_ids = [
        tid for tok, tid in char_vocab.items()
        if tok.startswith("[") and tok.endswith("]")
    ]
    collator = DualPhysicalDegradationCollator(
        mask_token_id=char_vocab["[MASK]"],
        pad_token_id=char_vocab["[PAD]"],
        unk_word_id=word_vocab.get("[UNK_WORD]", 0),
        vocab_char_size=len(char_vocab),
        special_token_ids=special_ids,
        mlm_prob=0.12,
        max_span=3,
        edge_prob=0.15,
        add_random_gaps=False,
        gap_token_id=char_vocab["[GAP]"],
        gap_prob=0.02,
        gap_span_min=1,
        gap_span_max=6,
        max_gaps=2,
    )

    # ========================================================================
    # ПАРАМЕТРЫ ТРЕНИРОВКИ
    # ========================================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = output_dir / "checkpoints"
    log_path = output_dir / f"training_log_{timestamp}.json"

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        overwrite_output_dir=True,
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
        report_to=[],  # Отключаем wandb/tensorboard
        load_best_model_at_end=True,
        metric_for_best_model="top1_accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        seed=args.seed,
    )

    # ========================================================================
    # ТРЕЙНЕР
    # ========================================================================

    log.info("Creating trainer...")
    trainer = LoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test_a"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        log_path=log_path,
    )

    # ========================================================================
    # ТРЕНИРОВКА
    # ========================================================================

    log.info("Starting training...")
    train_result = trainer.train()

    log.info("Training completed!")
    log.info(f"  Final training loss: {train_result.training_loss:.4f}")

    # Сохраняем логи
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.log_history, f, ensure_ascii=False, indent=2)
    log.info(f"Saved training logs: {log_path}")

    # ========================================================================
    # ФИНАЛЬНАЯ ОЦЕНКА
    # ========================================================================

    # Отключаем GAPы на валидации
    if hasattr(trainer.data_collator, "add_random_gaps"):
        trainer.data_collator.add_random_gaps = False

    log.info("Computing final metrics on test_a...")
    eval_metrics = trainer.evaluate()

    # Сохраняем метрики валидации
    metrics_path = output_dir / f"eval_metrics_{timestamp}.json"
    save_json(eval_metrics, metrics_path)

    for key, val in eval_metrics.items():
        if isinstance(val, (int, float)):
            log.info(f"  {key}: {val:.4f}")

    # ========================================================================
    # РЕПОРТ ПРЕДСКАЗАНИЙ
    # ========================================================================

    if args.report_test_b and "test_b" in dataset:
        log.info("Generating predictions report on test_b...")
        test_b_ds = dataset["test_b"]

        report_path = output_dir / f"predictions_report_{timestamp}.csv"
        report_metrics = generate_predictions_report(
            model=model,
            tokenizer=collator,
            char_vocab=char_vocab,
            word_vocab=word_vocab,
            dataset=test_b_ds,
            output_path=report_path,
            max_samples=args.max_report_samples,
        )

        # Сохраняем метрики репорта
        report_metrics_path = output_dir / f"predictions_report_metrics_{timestamp}.json"
        save_json(report_metrics, report_metrics_path)

        log.info("Predictions Report Summary:")
        for key, val in report_metrics.items():
            log.info(f"  {key}: {val:.4f}")

    # ========================================================================
    # СОХРАНЕНИЕ МОДЕЛИ
    # ========================================================================

    model_save_path = output_dir / "final_model"
    model_save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_save_path))
    log.info(f"Saved final model: {model_save_path}")

    # Сохраняем конфиг
    config_path = output_dir / "training_config.json"
    save_json(vars(args), config_path)

    # ========================================================================
    # ФИНАЛЬНЫЙ РЕПОРТ
    # ========================================================================

    final_report = {
        "timestamp": timestamp,
        "training_duration": train_result.training_loss,
        "train_metrics": eval_metrics,
        "args": vars(args),
    }

    if args.report_test_b and "test_b" in dataset:
        final_report["test_b_metrics"] = report_metrics

    report_path = output_dir / f"final_report_{timestamp}.json"
    save_json(final_report, report_path)

    log.info("=" * 80)
    log.info("Training completed successfully!")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Final report: {report_path}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()