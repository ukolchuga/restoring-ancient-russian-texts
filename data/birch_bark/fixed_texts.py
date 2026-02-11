import pandas as pd
import re
import os

INPUT_FILE = "gramoty_text_only.csv"

TARGET_COLS = ["original_text_spaced", "original_text_raw", "content"]


def correct_clean(text):
    if pd.isna(text):
        return text
    text = str(text)

    # 1. УДАЛЯЕМ ОШИБКИ ПИСЦА {текст}
    text = re.sub(r"\{[^}]+\}", "", text)

    # 2. ОСТАВЛЯЕМ ВОССТАНОВЛЕННЫЙ ТЕКСТ [текст], (текст)
    # Убираем скобки
    text = re.sub(r"[\[\]\(\)]", "", text)

    # 3. УДАЛЯЕМ ДЛИННЫЕ ТИРЕ (ЭТО ВАЖНО СДЕЛАТЬ СНАЧАЛА)
    # Если мы уберем их сейчас, они не будут мешать склейке слов
    # Ловим 2 и более любых тире подряд
    text = re.sub(r"[-‐‑–—−]{2,}", "", text)

    # 4. НОРМАЛИЗАЦИЯ ПРОБЕЛОВ
    # Теперь "купи- \n аи" превратится в "купи- аи"
    text = re.sub(r"\s+", " ", text)

    # 5. СКЛЕЙКА СЛОВ
    # Теперь между частями слова остался только один дефис и пробелы.
    # Склеиваем их.
    # Паттерн: Буква + (дефис) + (пробел) + Буква
    pattern = r"(\w)\s*[-‐‑–—−]\s*(\w)"

    # Делаем несколько проходов для надежности
    for _ in range(3):
        text = re.sub(pattern, r"\1\2", text)

    # 6. КОСМЕТИКА
    text = text.replace("…", "...")
    text = text.strip()

    # 7. ЗАЩИТА ОТ EXCEL
    if text.startswith(("+", "-", "=")) and not text.startswith("'"):
        text = "'" + text

    return text


def main():
    print(f"📂 Читаем файл: {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("Файл не найден.")
        return

    df = pd.read_csv(INPUT_FILE, dtype=str)

    # Обработка
    for col in TARGET_COLS:
        if col in df.columns:
            print(f"🔧 Чистим колонку: {col}...")
            df[col] = df[col].apply(correct_clean)

    # Проверка на твоем примере (купи- аи)
    text_col = "original_text_spaced"
    if text_col in df.columns:
        # Ищем, остались ли разрывы
        bad_hyphens = df[
            df[text_col].str.contains(r"(\w)\s*[-‐‑–—−]\s*(\w)", regex=True, na=False)
        ]
        print(f"Осталось разрывов: {len(bad_hyphens)}")

        if not bad_hyphens.empty:
            print("Пример оставшегося разрыва:", bad_hyphens[text_col].iloc[0])

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"✅ Готово: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
