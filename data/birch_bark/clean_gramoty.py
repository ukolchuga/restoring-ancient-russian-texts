import pandas as pd
import numpy as np
import re
import os

# ==========================================
# 📂 НАСТРОЙКИ ПУТЕЙ
# ==========================================
INPUT_CSV = "gramoty_text_only.csv"
OUTPUT_TXT = "gramoty_clean.txt"
OUTPUT_CSV = "gramoty_metadata.csv"


def unified_clean_pipeline(text):
    if pd.isna(text):
        return ""
    text = str(text)

    # 1. УБИРАЕМ МУСОР И ОШИБКИ
    text = re.sub(r"\{[^}]+\}", "", text)
    text = re.sub(
        r"(верхняя|нижняя|средняя)\s+часть\s+листа", "", text, flags=re.IGNORECASE
    )
    text = re.sub(r",[IVXLCDM]+\d*,\w+,\d+.*$", "", text)
    text = text.lstrip("\"' +`")

    # 2. РАСКРЫВАЕМ ВОССТАНОВЛЕННЫЕ БУКВЫ
    text = re.sub(r"[\[\]\(\)]", "", text)

    # 3. ПРЕВРАЩАЕМ ПРОПУСКИ В [GAP]
    text = re.sub(r"[-‐‑–—−]{2,}", " [GAP] ", text)
    text = re.sub(r"\.{3}|…|·-·", " [GAP] ", text)

    # 4. СКЛЕИВАЕМ РАЗОРВАННЫЕ СЛОВА
    pattern = r"(\w)\s*[-‐‑–—−]\s*(\w)"
    for _ in range(3):
        text = re.sub(pattern, r"\1\2", text)

    # 5. НОРМАЛИЗАЦИЯ И СХЛОПЫВАНИЕ
    text = re.sub(r"(\s*\[UNK\]\s*)+", " [GAP] ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 🔥 Убрано жесткое приклеивание [CTX_DAILY]
    return text


def count_tokens(text):
    # Считает [GAP], слова и пунктуацию как отдельные токены
    return len(re.findall(r"\[UNK\]|\w+|[^\w\s]", text))


def parse_dates(date_str):
    """Парсит строку даты, достает годы и вычисляет средний год и век"""
    if pd.isna(date_str):
        return pd.Series([np.nan, np.nan, np.nan, np.nan])

    nums = re.findall(r"\d{3,4}", str(date_str))

    if not nums:
        return pd.Series([np.nan, np.nan, np.nan, np.nan])

    nums = [int(n) for n in nums]
    start_year = nums[0]
    end_year = nums[-1]
    mean_year = int((start_year + end_year) / 2)
    century = (mean_year - 1) // 100 + 1

    return pd.Series([start_year, end_year, mean_year, century])


def main():
    print(f"📂 Читаем данные из: {INPUT_CSV}...")

    try:
        df = pd.read_csv(INPUT_CSV, dtype=str)
    except Exception as e:
        print(f"❌ Ошибка чтения CSV: {e}")
        return

    # 1. Очистка текста
    print("🔧 Чистим тексты...")
    text_col = next(
        (
            c
            for c in ["original_text_spaced", "original_text_raw", "content", "text"]
            if c in df.columns
        ),
        None,
    )

    if not text_col:
        print("❌ Ошибка: Не нашел колонку с текстом!")
        return

    df["clean"] = df[text_col].apply(unified_clean_pipeline)

    def is_valid_line(text):
        if not text or len(text) < 15:
            return False
        
        # Считаем только кириллические буквы (древнерусские и обычные)
        letters = re.findall(r"[а-яА-ЯёЁ\u0400-\u052F\uA640-\uA69F]", text)
        if len(letters) < 12: # Если меньше 12 букв - это обрывок
            return False
            
        words = text.split()
        gap_count = text.count("[GAP]")
        
        # Если [GAP] больше или равно половине "слов", строка слишком дырявая
        if gap_count > 1 and gap_count >= len(words) / 2:
            return False
            
        return True

    # Фильтруем мусор
    valid_rows = df[df["clean"].apply(is_valid_line)].copy()
    valid_rows["tokens"] = valid_rows["clean"].apply(count_tokens)

    # 2. Обработка дат
    print("⏳ Вычисляем века и средние годы...")
    if "date" in valid_rows.columns:
        valid_rows[["start_year", "end_year", "mean_year", "century"]] = valid_rows[
            "date"
        ].apply(parse_dates)

    # --- СОХРАНЕНИЕ ДЛЯ НЕЙРОСЕТИ (TXT) ---
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_rows["clean"].tolist()))

    # --- СОХРАНЕНИЕ ДЛЯ АНАЛИТИКИ (CSV) ---
    print("💾 Формируем таблицу метаданных...")

    potential_meta = [
        "id",
        "url",
        "city",
        "date",
        "start_year",
        "end_year",
        "mean_year",
        "century",
        "title",
        "content",
        "translation_ru",
    ]
    meta_cols = [col for col in potential_meta if col in valid_rows.columns]

    out_df = valid_rows[meta_cols + [text_col, "clean", "tokens"]].copy()
    out_df.rename(
        columns={text_col: "Original_Text", "clean": "Cleaned_Text"}, inplace=True
    )
    out_df.insert(0, "Line_in_TXT", range(1, len(out_df) + 1))

    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("🎉 ГОТОВО! Конвейер отработал.")
    print(f"📄 Чистые тексты: {OUTPUT_TXT}")
    print(f"📊 Метаданные: {OUTPUT_CSV}")
    print(f"Всего грамот: {len(valid_rows)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
