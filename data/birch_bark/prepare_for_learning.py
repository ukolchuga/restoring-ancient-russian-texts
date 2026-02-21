import pandas as pd
import re

INPUT_CSV = "gramoty_train_fixed.csv"
OUTPUT_TXT = "gramoty_clean.txt"
CSV_OUTPUT_FILE = "gramoty_metadata.csv"


def final_clean_gramoty(text):
    if not isinstance(text, str):
        return ""

    # Убираем пометки расположения текста
    text = re.sub(
        r"(верхняя|нижняя|средняя)\s+часть\s+листа", "", text, flags=re.IGNORECASE
    )

    # Убираем лишние кавычки в начале
    text = text.lstrip("\"' +`")

    # Унифицируем пропуски в токен [UNK]
    text = text.replace("...", " [UNK] ")
    text = text.replace("…", " [UNK] ")
    text = text.replace("·-·", " [UNK] ")

    # Схлопываем множественные [UNK], идущие подряд
    text = re.sub(r"(\s*\[UNK\]\s*)+", " [UNK] ", text)

    # Убираем технические хвосты (внутренние идентификаторы грамот)
    text = re.sub(r",[IVXLCDM]+\d*,\w+,\d+.*$", "", text)

    # Финальная чистка двойных пробелов
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():
    print(f"📜 Обрабатываем грамоты из {INPUT_CSV}...")

    try:
        df = pd.read_csv(INPUT_CSV, dtype=str)
    except Exception as e:
        print(f"❌ Ошибка чтения CSV: {e}")
        return

    text_col = next(
        (c for c in ["original_text_spaced", "text", "content"] if c in df.columns),
        None,
    )

    if not text_col:
        print("❌ Ошибка: Не нашел колонку с текстом!")
        return

    # 1. Очищаем тексты
    df["clean"] = df[text_col].apply(final_clean_gramoty)

    # 2. Фильтруем слишком короткие или пустые строки
    valid_rows = df[df["clean"].str.len() > 3].copy()

    # 3. Функция подсчета токенов (считает [UNK] как 1 токен)
    def count_tokens(text):
        return len(re.findall(r"\[UNK\]|\w+|[^\w\s]", text))

    # Применяем подсчет ко всем строкам
    valid_rows["tokens"] = valid_rows["clean"].apply(count_tokens)

    # 4. Сохраняем итоговый текстовый файл для обучения
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_rows["clean"].tolist()))

    # 5. Готовим данные для мета-таблицы
    meta_cols = []
    # Проверяем, есть ли нужные колонки в исходном файле
    for col in ["city", "year", "century"]:
        if col in valid_rows.columns:
            meta_cols.append(col)

    # Добавляем колонки с оригинальным и чистым текстом, а также токены
    columns_to_keep = meta_cols + [text_col, "clean", "tokens"]
    out_df = valid_rows[columns_to_keep].copy()

    # Переименовываем текстовые колонки для красоты в итоговой таблице
    out_df.rename(
        columns={text_col: "Original_Text", "clean": "Cleaned_Text"}, inplace=True
    )

    # Добавляем номер строки для привязки к файлу gramoty_clean.txt
    out_df.insert(0, "Line_in_TXT", range(1, len(out_df) + 1))

    # 6. Сохраняем в CSV
    out_df.to_csv(CSV_OUTPUT_FILE, index=False, encoding="utf-8")

    total_lines = len(valid_rows)
    total_tokens = valid_rows["tokens"].sum()

    print("\n" + "=" * 60)
    print("🎉 Готово! Грамоты вычищены, пропуски заменены на [UNK].")
    print(f"📄 Тексты для обучения сохранены в: {OUTPUT_TXT}")
    print(f"📊 Расширенная статистика сохранена в: {CSV_OUTPUT_FILE}")
    print(f"Всего грамот/строк: {total_lines}")
    print(f"Всего токенов (с учетом [UNK]): {total_tokens}")
    print("=" * 60)


if __name__ == "__main__":
    main()
