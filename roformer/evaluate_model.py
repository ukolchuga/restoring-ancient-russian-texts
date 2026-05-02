import json
import re

import evaluate
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, RoFormerForMaskedLM


def evaluate_test_a(model, tokenizer, test_a_path, device, batch_size=8):
    """
    Test A: Считаем Perplexity на чистом тексте.
    Оценивает базовое знание древнерусского языка.
    """
    print("\n--- Запуск Test A (Perplexity) ---")
    with open(test_a_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    encodings = tokenizer(
        lines, truncation=True, padding=True, max_length=512, return_tensors="pt"
    )
    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"], encodings["attention_mask"]
    )
    loader = DataLoader(dataset, batch_size=batch_size)

    total_loss = 0.0
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Оценка Test A"):
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            total_loss += outputs.loss.item() * input_ids.size(0)

    avg_loss = total_loss / len(lines)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print(f"✅ Test A Loss: {avg_loss:.4f}")
    print(f"✅ Test A Perplexity: {perplexity:.4f}")
    return perplexity


def evaluate_test_b(model, tokenizer, test_b_path, device):
    """
    Test B: Восстановление лакун.
    Считаем CER (Character Error Rate) и Exact Match (EM).
    """
    print("\n--- Запуск Test B (Lacunae Restoration) ---")
    cer_metric = evaluate.load("cer")

    exact_matches = 0
    total_samples = 0
    predictions_list = []
    references_list = []

    with open(test_b_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    model.eval()

    for record in tqdm(records, desc="Оценка Test B"):
        orig_text = record["original"]
        target_text = record["target"]

        # 1. Токенизируем target_text и получаем координаты (offsets) токенов
        encoded = tokenizer(
            target_text, return_offsets_mapping=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"][0].clone()
        offsets = encoded["offset_mapping"][0]

        # 2. Ищем спаны (координаты символов), которые были в скобках
        spans_to_mask = []
        orig_idx = 0

        # Регулярка игнорирует системные теги [GAP], [CTX_*] и ищет только наши лакуны
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

        # 3. Определяем BPE-токены, пересекающиеся с лакунами
        mask_token_indices = []
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:  # Пропускаем спецтокены (CLS, SEP)
                continue
            for span_start, span_end in spans_to_mask:
                if max(tok_start, span_start) < min(tok_end, span_end):
                    mask_token_indices.append(i)
                    break

        if not mask_token_indices:
            continue

        # 4. Маскируем нужные токены
        masked_input_ids = input_ids.clone()
        for idx in mask_token_indices:
            masked_input_ids[idx] = tokenizer.mask_token_id

        # 5. Инференс
        masked_input_ids = masked_input_ids.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(masked_input_ids)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu()

        # 6. Подставляем предсказанные токены на места масок
        final_ids = input_ids.clone()
        for idx in mask_token_indices:
            final_ids[idx] = predictions[idx]

        predicted_text = tokenizer.decode(final_ids, skip_special_tokens=True)

        predictions_list.append(predicted_text)
        references_list.append(target_text)

        if predicted_text == target_text:
            exact_matches += 1
        total_samples += 1

    # 7. Считаем метрики
    if total_samples > 0:
        em_score = (exact_matches / total_samples) * 100
        cer_score = (
            cer_metric.compute(predictions=predictions_list, references=references_list)
            * 100
        )

        print(f"✅ Обработано лакун: {total_samples}")
        print(f"✅ Exact Match (EM): {em_score:.2f}%")
        print(f"✅ Character Error Rate (CER): {cer_score:.2f}%")
    else:
        print("❌ Не найдено валидных лакун для тестирования.")


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
