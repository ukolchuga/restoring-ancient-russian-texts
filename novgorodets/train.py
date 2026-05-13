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
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, TrainerCallback

from config import DualBertConfig
from model import DualBertForMaskedLM
from collator import DualPhysicalDegradationCollator
from build_char_tokenizer import SPECIAL_TOKENS
import unicodedata

ALLOWED_PRED_IDS: set[int] | None = None

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
    """Подготавливает logits для расчета top-k метрик с учётом ALLOWED_PRED_IDS."""
    if isinstance(logits, tuple):
        logits = logits[0]  # [batch, seq_len, vocab]

    # Если фильтр не задан — поведение прежнее (берём topk по всем токенам)
    if ALLOWED_PRED_IDS is None:
        return torch.topk(logits, k=5, dim=-1).indices

    # Маскируем неподходящие id: ставим очень маленькие логиты
    vocab_size = logits.size(-1)
    allowed_mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    allowed_idxs = torch.tensor(list(ALLOWED_PRED_IDS), dtype=torch.long, device=logits.device)
    allowed_mask[allowed_idxs] = True

    # делаем копию logits и зануляем (вытесняем) дисаллоуед
    masked_logits = logits.clone()
    masked_logits[..., ~allowed_mask] = -1e9

    return torch.topk(masked_logits, k=5, dim=-1).indices

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
# ДОПОЛНИТЕЛЬНЫЙ ПРОГОН НА ТЕСТЕ B
# ============================================================================

class TestBEvalCallback(TrainerCallback):
    """
    Callback, который при каждом обычном evaluate (на test_a) дополнительно
    прогоняет evaluate на test_b и сохраняет метрики в output_dir.
    Защищён от рекурсии (чтобы не зациклиться).
    """

    def __init__(self, test_b_dataset, output_dir: Path, max_samples: int | None = None):
        self.test_b_dataset = test_b_dataset
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        self._in_eval = False

    def on_evaluate(self, args, state, control, **kwargs):
        # trainer приходит в kwargs
        trainer = kwargs.get("trainer")
        if trainer is None:
            return

        # Защита от рекурсивного вызова (evaluate -> on_evaluate -> evaluate -> ...)
        if self._in_eval:
            return

        if self.test_b_dataset is None:
            return

        # Запускаем дополнительную оценку
        try:
            self._in_eval = True
            # при желании ограничиваем число сэмплов для ускорения
            ds = self.test_b_dataset
            if self.max_samples is not None and hasattr(ds, "select"):
                n = min(self.max_samples, len(ds))
                ds = ds.select(range(n))

            metrics = trainer.evaluate(eval_dataset=ds)

            # сохраняем отдельно с шагом (если есть)
            step = getattr(state, "global_step", None) or "final"
            fname = self.output_dir / f"eval_metrics_test_b_step{step}.json"

            # save_json определён выше в файле
            save_json(metrics, fname)
            log.info(f"Saved Test B metrics: {fname}")

        finally:
            self._in_eval = False

# ============================================================================
# РЕПОРТ ПРЕДСКАЗАНИЙ
# ============================================================================

def generate_predictions_report(
    model,
    char_vocab: dict,
    dataset,
    output_path: Path,
    max_samples: int = 100,
    k_values: tuple = (1, 3, 5),
    device: Optional[torch.device] = None,
    collator=None,
    batch_size: int = 8,
    context_window: int = 20,  # количество символов до/после
) -> dict:
    """
    Универсальный репорт для test_a и test_b.

    - test_b: в dataset уже есть labels
    - test_a: labels нет, поэтому нужен collator, который создаст маскирование

    context_window: количество символов слева и справа для контекста
    """
    if device is None:
        device = next(model.parameters()).device

    id_to_char = {int(v): k for k, v in char_vocab.items()}

    def decode_one_token(token_id: int) -> str:
        return id_to_char.get(int(token_id), "[UNK]")

    def _mask_pred_logits_for_allowed(pred_logits: torch.Tensor) -> torch.Tensor:
        """Возвращает предлоги, где запрещённые id выставлены в -inf (т.е. не будут выбраны)."""
        if ALLOWED_PRED_IDS is None:
            return pred_logits
        vocab_size = pred_logits.size(-1)
        allowed_mask = torch.zeros(vocab_size, dtype=torch.bool, device=pred_logits.device)
        allowed_idxs = torch.tensor(list(ALLOWED_PRED_IDS), dtype=torch.long, device=pred_logits.device)
        allowed_mask[allowed_idxs] = True
        masked = pred_logits.clone()
        masked[..., ~allowed_mask] = -1e9
        return masked

    def get_context(input_ids: list, pos: int, context_window: int = 20) -> str:
        """Извлекает текстовый контекст вокруг позиции с правильным выделением."""
        start = max(0, pos - context_window)
        end = min(len(input_ids), pos + context_window + 1)

        # Декодируем части отдельно для точного позиционирования
        before = "".join(decode_one_token(int(cid)) for cid in input_ids[start:pos])
        target = decode_one_token(int(input_ids[pos]))
        after = "".join(decode_one_token(int(cid)) for cid in input_ids[pos+1:end])

        return before + ">>>" + target + "<<<" + after

    model.eval()
    rows = []
    hit_accum = {f"hit@{k}": 0 for k in k_values}
    correct = 0
    used = 0
    top_k_max = max(k_values)

    total_samples = min(len(dataset), max_samples)
    log.info(f"Generating predictions report for {total_samples} samples...")

    # --- test_b: labels уже есть ---
    if "labels" in dataset.column_names:
        for i, sample in enumerate(dataset.select(range(total_samples))):
            if i % 20 == 0:
                log.info(f"  Processing {i}/{total_samples}...")

            input_ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device)
            word_ids = torch.tensor(sample["word_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long, device=device)
            labels = sample.get("labels", None)

            if labels is None or -100 not in labels:
                continue

            mask_positions = [j for j, l in enumerate(labels) if l != -100]
            if not mask_positions:
                continue

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids.unsqueeze(0),
                    word_ids=word_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )
                logits = outputs.logits[0]

            input_ids_list = input_ids.cpu().numpy().tolist()

            for pos in mask_positions:
                true_id = int(labels[pos])
                if true_id < 0:
                    continue

                pred_logits = logits[pos]
                pred_logits_masked = _mask_pred_logits_for_allowed(pred_logits)
                top_ids = torch.topk(pred_logits_masked, k=min(top_k_max, len(char_vocab))).indices.cpu().numpy()

                pred_id = int(top_ids[0])
                true_char = decode_one_token(true_id)
                pred_char = decode_one_token(pred_id)

                if pred_char.startswith("[") or true_char.startswith("["):
                    continue

                used += 1
                is_correct = pred_id == true_id
                if is_correct:
                    correct += 1

                probs = torch.softmax(pred_logits, dim=-1)
                top1_prob = float(probs[pred_id].item())

                true_rank = next(
                    (r + 1 for r, tid in enumerate(top_ids) if int(tid) == true_id),
                    None,
                )
                top_chars = [decode_one_token(int(tid)) for tid in top_ids[:5]]

                # Контекст вокруг позиции
                context = get_context(input_ids_list, pos, context_window)

                for k in k_values:
                    if any(decode_one_token(int(tid)) == true_char for tid in top_ids[:k]):
                        hit_accum[f"hit@{k}"] += 1

                rows.append({
                    "sample_idx": i,
                    "position": pos,
                    "context": context,
                    "true_char": true_char,
                    "pred_char": pred_char,
                    "is_correct": is_correct,
                    "true_rank": true_rank,
                    "top1_prob": round(top1_prob, 4),
                    "top5_preds": "|".join(top_chars),
                })

    # --- test_a: labels создаём через collator ---
    else:
        if collator is None:
            raise ValueError("For dataset without labels (test_a), collator must be provided.")

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_raw = [dataset[j] for j in range(start, end)]

            if start % (batch_size * 10) == 0:
                log.info(f"  Processing {start}/{total_samples}...")

            batch = collator(batch_raw)

            input_ids = batch["input_ids"].to(device)
            word_ids = batch["word_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"]

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    word_ids=word_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits

            for b, labels in enumerate(labels_batch):
                labels = labels.tolist() if isinstance(labels, torch.Tensor) else labels
                mask_positions = [j for j, l in enumerate(labels) if l != -100]
                if not mask_positions:
                    continue

                sample_input_ids = input_ids[b].cpu().numpy().tolist()
                sample_logits = logits[b]

                for pos in mask_positions:
                    true_id = int(labels[pos])
                    if true_id < 0:
                        continue

                    pred_logits = sample_logits[pos]
                    pred_logits_masked = _mask_pred_logits_for_allowed(pred_logits)
                    top_ids = torch.topk(pred_logits_masked, k=min(top_k_max, len(char_vocab))).indices.cpu().numpy()

                    pred_id = int(top_ids[0])
                    true_char = decode_one_token(true_id)
                    pred_char = decode_one_token(pred_id)

                    if pred_char.startswith("[") or true_char.startswith("["):
                        continue

                    used += 1
                    is_correct = pred_id == true_id
                    if is_correct:
                        correct += 1

                    probs = torch.softmax(pred_logits, dim=-1)
                    top1_prob = float(probs[pred_id].item())

                    true_rank = next(
                        (r + 1 for r, tid in enumerate(top_ids) if int(tid) == true_id),
                        None,
                    )
                    top_chars = [decode_one_token(int(tid)) for tid in top_ids[:5]]

                    # Контекст вокруг позиции
                    context = get_context(sample_input_ids, pos, context_window)

                    for k in k_values:
                        if any(decode_one_token(int(tid)) == true_char for tid in top_ids[:k]):
                            hit_accum[f"hit@{k}"] += 1

                    rows.append({
                        "sample_idx": start + b,
                        "position": pos,
                        "context": context,
                        "true_char": true_char,
                        "pred_char": pred_char,
                        "is_correct": is_correct,
                        "true_rank": true_rank,
                        "top1_prob": round(top1_prob, 4),
                        "top5_preds": "|".join(top_chars),
                    })

    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        log.info(f"Saved predictions report: {output_path}")

    metrics = {
        "total_predictions": used,
        "correct": correct,
        "accuracy": round(correct / used, 4) if used > 0 else 0.0,
        **{k: round(v / used, 4) if used > 0 else 0.0 for k, v in hit_accum.items()},
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=400)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Репорт
    parser.add_argument(
        "--report_test_a",
        action="store_true",
        help="Репорт на test_a",
    )
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

    # ----- формируем allowed ids: разрешаем предсказывать буквы и пробелы (категория Zs)
    global ALLOWED_PRED_IDS
    ALLOWED_PRED_IDS = {
        int(v) for k, v in char_vocab.items()
        if len(k) == 1 and unicodedata.category(k) in ("Ll", "Lu", "Lo", "Zs")
    }

    special_ids = [char_vocab[tok] for tok in SPECIAL_TOKENS if tok in char_vocab]

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
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=3,
        logging_steps=100,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,

        # fp16=args.fp16,
        fp16=False,
        # bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        max_grad_norm=1.0, #  gradient clipping

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

    callbacks = []

    # callback для автоматического evaluate на test_b
    if "test_b" in dataset:
        tb_cb = TestBEvalCallback(
            test_b_dataset=dataset["test_b"],
            output_dir=output_dir,
            max_samples=args.max_report_samples
        )
        callbacks.append(tb_cb)

    trainer = LoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test_a"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        log_path=log_path,
        callbacks=callbacks if callbacks else None,
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

    report_specs = [
        (
            args.report_test_a,
            "test_a",
            dataset["test_a"] if "test_a" in dataset else None,
            output_dir / f"predictions_report_test_a_{timestamp}.csv",
            collator,
        ),
        (
            args.report_test_b,
            "test_b",
            dataset["test_b"] if "test_b" in dataset else None,
            output_dir / f"predictions_report_test_b_{timestamp}.csv",
            None,
        ),
    ]

    report_metrics = {}

    for enabled, name, ds, report_path, report_collator in report_specs:
        if not enabled or ds is None:
            continue

        log.info(f"Generating predictions report on {name}...")
        metrics = generate_predictions_report(
            model=model,
            char_vocab=char_vocab,
            dataset=ds,
            output_path=report_path,
            max_samples=args.max_report_samples,
            collator=report_collator,
            batch_size=8,
        )

        metrics_path = output_dir / f"predictions_report_{name}_metrics_{timestamp}.json"
        save_json(metrics, metrics_path)

        log.info(f"{name} report summary:")
        for key, val in metrics.items():
            log.info(f"  {key}: {val:.4f}")

        report_metrics[name] = metrics

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

    if "test_a" in report_metrics:
        final_report["test_a_metrics"] = report_metrics["test_a"]

    if "test_b" in report_metrics:
        final_report["test_b_metrics"] = report_metrics["test_b"]

    report_path = output_dir / f"final_report_{timestamp}.json"
    save_json(final_report, report_path)

    log.info("=" * 80)
    log.info("Training completed successfully!")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Final report: {report_path}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()