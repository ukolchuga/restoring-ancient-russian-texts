import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import re
import csv

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
    "Комментарий": "comment"
}

# Все колонки в итоговом файле
COLUMNS = ["id", "url"] + list(FIELDS_MAP.values())

def clean_val(text):
    if not text: return ""
    # Убираем лишние пробелы, кавычки и странные символы
    text = text.replace('"', "'") # Заменяем двойные кавычки на одинарные, чтобы не ломать CSV
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_inscription(ins_id):
    url = f"{BASE_URL}{ins_id}"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        if "Надпись не найдена" in soup.text:
            return None

        table = soup.find("table", class_="table-bordered")
        if not table:
            return None

        # Инициализируем словарь пустыми значениями
        row_data = {col: "" for col in COLUMNS}
        row_data["id"] = ins_id
        row_data["url"] = url
        
        for tr in table.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if th and td:
                raw_key = th.get_text(strip=True).replace("см. Условные обозначения", "").strip()
                
                if raw_key in FIELDS_MAP:
                    col_name = FIELDS_MAP[raw_key]
                    
                    if raw_key == "Текст":
                        text_div = td.find("div", class_="eomr-text-wrapper")
                        if text_div:
                            for br in text_div.find_all("br"):
                                br.replace_with(" ")
                            val = text_div.get_text(strip=True)
                        else:
                            val = td.get_text(strip=True)
                    else:
                        # Очистка от ссылок на литературу внутри текста
                        for a in td.find_all("a"):
                            if "bibliography" in a.get('href', ''):
                                a.decompose()
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
    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

    new_data_count = 0
    # Идем по порядку от 1 до 1215
    for ins_id in range(1, MAX_ID + 1):
        print(f"[{ins_id}/{MAX_ID}] Fetching...", end=" ", flush=True)
        data = parse_inscription(ins_id)
        
        if data:
            with open(OUTPUT_CSV, 'a', encoding='utf-8-sig', newline='') as f:
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
