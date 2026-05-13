import json
import math
import re

import evaluate
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, RoFormerForMaskedLM

from roformer.collator import RoFormerPhysicalDegradationCollator


def evaluate_test_a(model, tokenizer, test_a_path, device, batch_size=8):
    """
    Test A: Считаем Perplexity на тексте, маскированном под физическую деградацию.
    Оценивает базовое знание языка и способность справляться со сложными лакунами (спаны, края).
    """
    print("\n--- Запуск Test A (Physical Degradation Perplexity) ---")
    with open(test_a_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Токенизируем
    encodings = tokenizer(lines, truncation=True, max_length=512)

    dataset = []
    for i in range(len(encodings["input_ids"])):
        dataset.append(
            {
                "input_ids": encodings["input_ids"][i],
                "attention_mask": encodings["attention_mask"][i],
            }
        )

    # ИСПОЛЬЗУЕМ ТВОЙ КАСТОМНЫЙ КОЛЛАТОР!
    # Выставляем add_random_gaps=False для теста, чтобы оценивать
    # только предсказание масок, а не угадывание, что было до [GAP]
    # (но если хочешь хардкора, можешь включить)
    data_collator = RoFormerPhysicalDegradationCollator(
        tokenizer=tokenizer,
        mlm_prob=0.15,
        max_span=3,
        edge_prob=0.1,
        add_random_gaps=False,  # Для стабильного подсчета лосса лучше выключить [GAP] в тесте
    )

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    total_loss = 0.0
    total_batches = 0
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Оценка Test A"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print(f"✅ Test A Loss: {avg_loss:.4f}")
    print(f"✅ Test A Perplexity: {perplexity:.4f}")
    return perplexity


def evaluate_test_b(model, tokenizer, test_b_path, device, top_k=5):
    """
    Test B: Восстановление лакун с использованием Sequential Decoding (Beam Search).
    Считаем CER, Exact Match @ 1 и Exact Match @ Top-5.
    """
    print(f"\n--- Запуск Test B (Sequential Decoding, Top-K={top_k}) ---")
    cer_metric = evaluate.load("cer")

    exact_matches_top1 = 0
    exact_matches_top5 = 0
    total_samples = 0

    predictions_list = []  # Для CER берем только лучший вариант
    references_list = []

    with open(test_b_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    model.eval()

    for record in tqdm(records, desc="Оценка Test B"):
        orig_text = record["original"]
        target_text = record["target"]

        # 1. Токенизация и поиск спанов
        encoded = tokenizer(
            target_text, return_offsets_mapping=True, return_tensors="pt"
        )
        base_input_ids = encoded["input_ids"][0].clone()
        offsets = encoded["offset_mapping"][0]

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

        # 2. Поиск токенов для маскирования
        mask_token_indices = []
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:
                continue
            for span_start, span_end in spans_to_mask:
                if max(tok_start, span_start) < min(tok_end, span_end):
                    mask_token_indices.append(i)
                    break

        if not mask_token_indices:
            continue

        # 3. Подготовка стартового состояния (маскируем токены)
        masked_input_ids = base_input_ids.clone()
        for idx in mask_token_indices:
            masked_input_ids[idx] = tokenizer.mask_token_id

        # 4. EASY-FIRST SEQUENTIAL DECODING
        current_states = [{"input_ids": masked_input_ids, "log_prob": 0.0}]
        unfilled_masks = mask_token_indices.copy()

        with torch.no_grad():
            while unfilled_masks:
                # ШАГ 1: Ищем самую "простую" маску, используя лучшую текущую гипотезу
                best_state_ids = current_states[0]["input_ids"].unsqueeze(0).to(device)
                outputs = model(input_ids=best_state_ids)
                logits = outputs.logits[0]

                best_mask_idx = None
                highest_prob = -1.0

                for m_idx in unfilled_masks:
                    probs = torch.nn.functional.softmax(logits[m_idx], dim=-1)
                    max_prob = torch.max(probs).item()
                    if max_prob > highest_prob:
                        highest_prob = max_prob
                        best_mask_idx = m_idx

                # Удаляем найденную маску из списка незаполненных
                unfilled_masks.remove(best_mask_idx)

                # ШАГ 2: Батч-прогон и ветвление лучей (Beam Search) для выбранной маски
                new_candidates = []
                batch_inputs = torch.stack(
                    [state["input_ids"] for state in current_states]
                ).to(device)
                outputs = model(input_ids=batch_inputs)

                # Получаем логиты только для выбранной (самой простой) маски
                mask_logits = outputs.logits[:, best_mask_idx, :]
                mask_probs = torch.nn.functional.softmax(mask_logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(mask_probs, top_k, dim=-1)

                for state_idx, state in enumerate(current_states):
                    for i in range(top_k):
                        token_id = top_k_indices[state_idx, i].item()
                        prob = top_k_probs[state_idx, i].item()

                        new_input_ids = state["input_ids"].clone()
                        new_input_ids[best_mask_idx] = token_id
                        new_log_prob = state["log_prob"] + math.log(max(prob, 1e-9))

                        new_candidates.append(
                            {"input_ids": new_input_ids, "log_prob": new_log_prob}
                        )

                # Оставляем только лучшие Top-K гипотез
                current_states = sorted(
                    new_candidates, key=lambda x: x["log_prob"], reverse=True
                )[:top_k]

        # 5. Оценка результатов
        # Декодируем чистый таргет для точного сравнения
        clean_target = tokenizer.decode(base_input_ids, skip_special_tokens=True)

        # Декодируем сгенерированные гипотезы
        generated_texts = []
        for state in current_states:
            gen_text = tokenizer.decode(state["input_ids"], skip_special_tokens=True)
            generated_texts.append(gen_text)

        best_prediction = generated_texts[0]

        predictions_list.append(best_prediction)
        references_list.append(clean_target)

        # Считаем Exact Match
        if clean_target == best_prediction:
            exact_matches_top1 += 1

        if clean_target in generated_texts:
            exact_matches_top5 += 1

        total_samples += 1

    # 6. Итоговые метрики
    if total_samples > 0:
        em1_score = (exact_matches_top1 / total_samples) * 100
        em5_score = (exact_matches_top5 / total_samples) * 100
        cer_score = (
            cer_metric.compute(predictions=predictions_list, references=references_list)
            * 100
        )

        print(f"✅ Обработано лакун: {total_samples}")
        print(
            f"✅ Exact Match @ 1: {em1_score:.2f}% (Идеальное совпадение лучшего варианта)"
        )
        print(
            f"✅ Exact Match @ {top_k}: {em5_score:.2f}% (Правильный ответ есть в топ-{top_k})"
        )
        print(f"✅ Character Error Rate (CER): {cer_score:.2f}% (Чем ниже, тем лучше)")

        return {"em_1": em1_score, "em_5": em5_score, "cer": cer_score}
    else:
        print("❌ Не найдено валидных лакун для тестирования.")
        return {}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # TODO: Укажи путь к своей обученной модели
    MODEL_PATH = "./roformer-ancient-rus"
    TEST_A_PATH = "splits/test_a.txt"
    TEST_B_PATH = "splits/test_b.jsonl"

    print("Загрузка модели и токенизатора...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = RoFormerForMaskedLM.from_pretrained(MODEL_PATH).to(device)

        evaluate_test_a(model, tokenizer, TEST_A_PATH, device)
        evaluate_test_b(model, tokenizer, TEST_B_PATH, device)
    except Exception as e:
        print(
            f"Ошибка загрузки модели. Убедись, что путь '{MODEL_PATH}' правильный. Ошибка: {e}"
        )
