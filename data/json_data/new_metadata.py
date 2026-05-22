import csv
import json
import os
import re

JSON_FILE = "DIACU_1.0.json"  # Путь к твоему JSON
OUTPUT_CSV = "diacu_metadata_2.csv"


def sanitize_filename(title: str) -> str:
    """Очистка имени файла для связи txt и csv"""
    clean = re.sub(r"[^\w\s-]", "", title).strip()
    return re.sub(r"[\s-]+", "_", clean)


def parse_century_to_years(century_str: str):
    """
    Мощный парсер дат, заточенный под твои 43 уникальных варианта.
    """
    if not century_str or century_str.strip() in ["", "not given"]:
        return "", "", ""

    c_str = century_str.lower()

    # 1. Заменяем римские цифры и частые опечатки
    c_str = c_str.replace("xvi", "16").replace("16t ", "16th ")

    # 2. Ищем все числа в строке
    nums = [int(n) for n in re.findall(r"\d+", c_str)]

    if not nums:
        return "", "", ""

    # 3. Находим самый ранний и самый поздний век в строке
    start_cent = min(nums)
    end_cent = max(nums)

    # 4. Переводим века в годы (например, 14 век = 1301 - 1400)
    y_min = (start_cent - 1) * 100 + 1
    y_max = end_cent * 100

    # 5. Уточняем диапазоны, если век всего один (или если это точный период)
    if start_cent == end_cent:
        if "first half" in c_str:
            y_max = y_min + 49
        elif "second half" in c_str:
            y_min = y_min + 50
        elif "first quarter" in c_str:
            y_max = y_min + 24
        elif "second quarter" in c_str:
            y_min = y_min + 25
            y_max = y_min + 49
        elif "third quarter" in c_str:
            y_min = y_min + 50
            y_max = y_min + 74
        elif "last quarter" in c_str or "fourth quarter" in c_str:
            y_min = y_max - 24
        elif "early" in c_str or "beginning" in c_str:
            y_max = y_min + 33
        elif "mid" in c_str or "middle" in c_str:
            y_min = y_min + 33
            y_max = y_min + 33
        elif "late" in c_str or "end of" in c_str:
            y_min = y_max - 33

    # Если есть составная фраза вроде "End of the 15th – first half of the 16th"
    # y_min и y_max просто возьмут границы (1401 - 1600), что даст нормальное среднее

    target = int((y_min + y_max) / 2)
    return y_min, y_max, target


def main():
    if not os.path.exists(JSON_FILE):
        print(f"❌ Файл {JSON_FILE} не найден.")
        return

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data.get("Documents", [])
    print(f"⏳ Найдено {len(docs)} документов. Строим CSV...")

    with open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as csv_file:
        # Оставляем сырые Epoch и Language
        fieldnames = [
            "Filename",
            "Title",
            "Raw_Language",
            "Raw_Epoch",
            "Original_Date_Str",
            "Y_min",
            "Y_max",
            "Target_Year",
            "Source",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, doc in enumerate(docs):
            title = doc.get("Title", "") or doc.get("Original_title", f"doc_{idx}")
            filename = f"{idx + 1:03d}_{sanitize_filename(title)}.txt"

            raw_lang = doc.get("Language", "")
            raw_epoch = doc.get("Epoch", "")

            century_raw = doc.get("Century", "")
            y_min, y_max, target = parse_century_to_years(century_raw)

            writer.writerow(
                {
                    "Filename": filename,
                    "Title": title,
                    "Raw_Language": raw_lang,
                    "Raw_Epoch": raw_epoch,
                    "Original_Date_Str": century_raw,
                    "Y_min": y_min,
                    "Y_max": y_max,
                    "Target_Year": target,
                    "Source": doc.get("Source", ""),
                }
            )

    print(f"✅ Успех! Файл сохранен как {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
