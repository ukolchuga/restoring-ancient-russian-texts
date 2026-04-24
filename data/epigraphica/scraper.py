import csv
import os
import re
import time

import requests
from bs4 import BeautifulSoup

# Настройки
BASE_URL = "https://epigraphica.ru/epigraphy/inscription/show/"
OUTPUT_DIR = "data/epigraphica"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "epigraphica_full_data.csv")
MAX_ID = 1215
DELAY = 0.3

# Маппинг полей: ключ на сайте -> имя колонки в CSV
FIELDS_MAP = {
    "Датировка": "date",
    "Содержание": "content",
    "Носитель": "carrier",
    "Категория носителя": "carrier_category",
    "Место находки": "place",
    "Текст": "text",
    "Реконструкция": "reconstruction",
    "Перевод": "translation",
    "Алфавит": "alphabet",
    "Способ нанесения": "writing_method",
    "Расположение на носителе": "position",
    "Состояние сохранности": "preservation",
    "Датировка по палеографии и языку": "date_details",
    "Обоснование датировки": "date_justification",
    "Комментарий": "comment",
}

# Все колонки в итоговом файле
COLUMNS = ["id", "url"] + list(FIELDS_MAP.values())


def clean_val(text):
    if not text:
        return ""

    # 1. Жестко убиваем все переносы строк, заменяя их на пробелы
    text = text.replace("\n", " ").replace("\r", " ")

    # 2. Убираем "призрачные скобки" (пустые скобки () или (  ), оставшиеся после удаления ссылок)
    text = re.sub(r"\(\s*\)", "", text)

    # 3. Схлопываем идущие подряд пробелы и табы
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def parse_inscription(ins_id):
    url = f"{BASE_URL}{ins_id}"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        if "Надпись не найдена" in soup.text:
            return None

        table = soup.find("table", class_="table-bordered")
        if not table:
            return None

        row_data = {col: "" for col in COLUMNS}
        row_data["id"] = ins_id
        row_data["url"] = url

        for tr in table.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")

            if th and td:
                raw_key = (
                    th.get_text(strip=True)
                    .replace("см. Условные обозначения", "")
                    .strip()
                )

                if raw_key in FIELDS_MAP:
                    col_name = FIELDS_MAP[raw_key]

                    # Очистка ссылок на литературу
                    for a in td.find_all("a"):
                        if a.get("href") and "bibliography" in a.get("href"):
                            a.decompose()

                    # ВАЖНО: Используем separator=" ", чтобы склеить текст через пробел, а не через Enter
                    val = td.get_text(separator=" ", strip=True)
                    row_data[col_name] = clean_val(val)

        return row_data
    except Exception as e:
        print(f"Error fetching ID {ins_id}: {e}")
        return None


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Если файл существует, удаляем его, чтобы начать "чисто" (как просил пользователь)
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        print("Existing file removed. Starting fresh.")

    # Создаем файл и пишем заголовки
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

    new_data_count = 0
    # Идем по порядку от 1 до 1215
    for ins_id in range(1, MAX_ID + 1):
        print(f"[{ins_id}/{MAX_ID}] Fetching...", end=" ", flush=True)
        data = parse_inscription(ins_id)

        if data:
            with open(OUTPUT_CSV, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writerow(data)

            print(f"SUCCESS: {data['content'][:50]}...")
            new_data_count += 1
        else:
            print("SKIP (Not found)")

        time.sleep(DELAY)

    print(f"\nDone! Saved {new_data_count} inscriptions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
