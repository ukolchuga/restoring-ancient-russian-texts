import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os

# --- НАСТРОЙКИ ---
START_ID = 1
END_ID = 1300  # Пройдемся по всем номерам
OUTPUT_FILE = 'gramoty_text_only.csv'

# Список слагов городов для перебора URL
CITY_SLUGS = [
    'novgorod', 'staraya-russa', 'torzhok', 'smolensk', 'pskov',
    'tver', 'moscow', 'staraya-ryazan', 'zvenigorod',
    'vitebsk', 'mstislavl', 'vologda', 'pereyaslavl-ryazansky'
]


def clean_text(text):
    if not text: return None
    # Убираем неразрывные пробелы и лишние переносы
    text = text.replace('\xa0', ' ').replace('\n', ' ')
    return " ".join(text.split())


def get_page_soup(doc_id):
    """
    Проверяет ВСЕ города для данного номера.
    Возвращает список кортежей: [(soup, url, slug), ...]
    """
    found_pages = []

    # Фразы, которые говорят о том, что страницы НЕТ, даже если код 200
    BAD_PHRASES = [
        "Документ не найден",
        "Oops! An Error Occurred",
        "404 Not Found",
        "Something is broken"
    ]

    for slug in CITY_SLUGS:
        url = f"https://gramoty.ru/birchbark/document/show/{slug}/{doc_id}/"
        try:
            r = requests.get(url, timeout=2)

            # Проверяем статус
            if r.status_code == 200:
                # Проверяем, не заглушка ли это
                page_text = r.text
                if not any(phrase in page_text for phrase in BAD_PHRASES):
                    soup = BeautifulSoup(r.content, 'html.parser')
                    found_pages.append((soup, url, slug))

        except Exception:
            continue

    return found_pages

def parse_gramota_html(soup, doc_id, url, city_slug):
    data = {
        'id': doc_id,
        'url': url,
        'city_slug': city_slug,
        'title': None,
        'city': None,
        'date': None,
        'content': None,
        'original_text_raw': None,
        'original_text_spaced': None,
        'translation_ru': None,
        'translation_en_1': None,
        'translation_en_2': None
    }

    # 1. ЗАГОЛОВОК
    h1 = soup.find('h1')
    if h1:
        for span in h1.find_all('span'):
            span.decompose()
        data['title'] = clean_text(h1.get_text())
    else:
        data['title'] = f"Грамота {doc_id}"

    # 2. ТАБЛИЦА
    table = soup.find('table', class_='mr-show-table')
    if table:
        rows = table.find_all('tr')
        en_trans_count = 0

        for row in rows:
            th = row.find('th')
            td = row.find('td')
            if not th or not td: continue

            header = th.get_text(strip=True).lower()

            if 'город' in header:
                data['city'] = clean_text(td.get_text())
            elif 'условная дата' in header:
                data['date'] = clean_text(td.get_text())
            elif 'содержание' in header:
                data['content'] = clean_text(td.get_text())

            # --- ТЕКСТ (ФИНАЛЬНАЯ ВЕРСИЯ ОЧИСТКИ) ---
            elif 'текст' in header:
                # Список фраз, которые надо удалить из текста грамоты
                garbage = [
                    "Внешняя сторона", "Внутренняя сторона",
                    "Сторона А", "Сторона Б", "Сторона В",
                    "Фрагмент А", "Фрагмент Б",
                    "Целый текст", "Верхняя часть", "Нижняя часть"
                ]

                # --- 1. RAW (Без пробелов) ---
                wrapper_raw = td.find('div', class_='original-text-wrapper without-spaces')
                if wrapper_raw:
                    parts = wrapper_raw.find_all('div', class_='original-text')
                    if parts:
                        raw_txt = " ".join([clean_text(p.get_text()) for p in parts])
                    else:
                        raw_txt = clean_text(wrapper_raw.get_text())

                    # Чистим мусор
                    if raw_txt:
                        for g in garbage:
                            raw_txt = raw_txt.replace(g, "")
                        data['original_text_raw'] = clean_text(raw_txt)
                else:
                    data['original_text_raw'] = None

                # --- 2. SPACED (С пробелами) ---
                wrapper_spaced = td.find('div', class_='original-text-wrapper with-spaces')
                if wrapper_spaced:
                    parts = wrapper_spaced.find_all('div', class_='original-text')
                    if parts:
                        spaced_txt = " ".join([clean_text(p.get_text()) for p in parts])
                    else:
                        spaced_txt = clean_text(wrapper_spaced.get_text())

                    # Чистим мусор
                    if spaced_txt:
                        for g in garbage:
                            spaced_txt = spaced_txt.replace(g, "")
                        data['original_text_spaced'] = clean_text(spaced_txt)
                else:
                    data['original_text_spaced'] = None

                # Фоллбек: если нет RAW, делаем из SPACED
                if not data['original_text_raw'] and data['original_text_spaced']:
                    data['original_text_raw'] = data['original_text_spaced'].replace(' ', '')

            # --- ПЕРЕВОДЫ ---
            elif 'русский перевод' in header:
                trans_div = td.find('div', class_='translated-text-wrapper')
                if trans_div:
                    val = clean_text(trans_div.get_text())
                    if val and "Перевода нет" not in val:
                        data['translation_ru'] = val

            elif 'english translation' in header:
                trans_div = td.find('div', class_='translated-text-wrapper')
                if trans_div:
                    text_en = clean_text(trans_div.get_text())
                    if text_en and "No translation available" not in text_en:
                        en_trans_count += 1
                        if en_trans_count == 1:
                            data['translation_en_1'] = text_en
                        elif en_trans_count == 2:
                            data['translation_en_2'] = text_en

    return data


# --- ЗАПУСК ---
print(f"Подготовка к сбору (ID {START_ID}-{END_ID})...")

all_records = []
current_start_id = START_ID

# 1. ПРОВЕРКА: Есть ли уже файл с данными?
if os.path.exists(OUTPUT_FILE):
    try:
        print(f"Найден существующий файл {OUTPUT_FILE}. Читаем...")
        df_existing = pd.read_csv(OUTPUT_FILE)

        if not df_existing.empty:
            # Находим максимальный ID, который уже скачали
            last_id = int(df_existing['id'].max())
            print(f"Последний записанный ID: {last_id}")

            if last_id < END_ID:
                current_start_id = last_id + 1
                # Загружаем старые данные обратно в память, чтобы не потерять их при перезаписи
                all_records = df_existing.to_dict('records')
                print(f"--> Продолжаем скачивание с ID {current_start_id}")
            else:
                print("Похоже, все данные уже скачаны!")
                current_start_id = END_ID + 1  # Чтобы цикл не запускался
        else:
            print("Файл пустой, начинаем с нуля.")
    except Exception as e:
        print(f"Ошибка чтения файла (начнем заново): {e}")

# 2. ОСНОВНОЙ ЦИКЛ
for i in range(current_start_id, END_ID + 1):
    pages = get_page_soup(i)

    found_any = False
    if pages:
        for soup, url, slug in pages:
            try:
                record = parse_gramota_html(soup, i, url, slug)
                record['unique_id'] = f"{slug}_{i}"

                # Добавляем в общий список
                all_records.append(record)

                has_text = "ЕСТЬ ТЕКСТ" if record['original_text_spaced'] else "ПУСТО"
                print(f"[+] {slug.upper()} №{i}: {record['date']} | {has_text}")
                found_any = True
            except Exception as e:
                print(f"[!] Ошибка обработки {url}: {e}")

    if not found_any:
        # Если ни в одном городе нет такого номера (например, пропущен),
        # всё равно печатаем, чтобы видеть прогресс
        print(f"[-] №{i} не найден нигде")

    # 3. СОХРАНЕНИЕ (каждые 10 штук для надежности)
    if i % 10 == 0:
        if all_records:
            df = pd.DataFrame(all_records)
            # Удаляем дубликаты на всякий случай
            df = df.drop_duplicates(subset=['unique_id'])
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            print(f"--- АВТОСОХРАНЕНИЕ: Всего {len(df)} записей ---")

    time.sleep(0.05)

# Финал
if all_records:
    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=['unique_id'])
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"Готово! Полный файл сохранен: {OUTPUT_FILE}")
else:
    print("Данные не найдены.")