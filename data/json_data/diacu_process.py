import os
import re
import unicodedata
import csv

# ПАРАМЕТРЫ
INPUT_FOLDER = "diacu_extracted"  # Папка с твоими сырыми txt файлами DIACU
OUTPUT_FOLDER = "diacu_cleaned_texts"  # Куда кладем чистые txt
CSV_OUTPUT_FILE = "diacu_statistics.csv"  # Файл для Google Таблиц

TITLO_RANGE = range(0x0483, 0x0488)  # Диапазон кириллических титлов

# Словарь для спасения слов с опечатками (заменяем латиницу на кириллицу)
REPLACEMENTS = {
    "A": "А",
    "a": "а",
    "B": "В",
    "E": "Е",
    "e": "е",
    "K": "К",
    "k": "к",
    "M": "М",
    "H": "Н",
    "O": "О",
    "o": "о",
    "P": "Р",
    "p": "р",
    "C": "С",
    "c": "с",
    "T": "Т",
    "y": "у",
    "Y": "У",
    "X": "Х",
    "x": "х",
    "R": "Я",
    "r": "я",
    "U": "У",
    "u": "у",
    "Z": "З",
    "z": "з",
    "W": "Ѡ",
    "w": "ѡ",
    "F": "Ф",
    "f": "ф",
    "S": "Ѕ",
    "s": "ѕ",
    "I": "І",
    "i": "і",
    "N": "Н",
    "n": "н",
}


def split_long_line(text, max_len=500):
    if len(text) <= max_len:
        return [text]
    parts = []
    while len(text) > max_len:
        split_idx = text.rfind(" ", 0, max_len)
        if split_idx == -1:
            split_idx = max_len
        parts.append(text[:split_idx].strip())
        text = text[split_idx:].strip()
    if text:
        parts.append(text)
    return parts


def clean_diacu_text(text):
    if not text:
        return ""

    # 1. Спасаем слова, заменяя очевидные латинские опечатки на кириллицу
    for lat, cyr in REPLACEMENTS.items():
        text = text.replace(lat, cyr)

    # 2. Восстанавливаем древнерусский знак тысячи (҂) вместо $ и @
    text = text.replace("$", "҂").replace("@", "҂")

    # 3. Очистка от технического мусора и XML-разметки
    text = text.replace("\x00", "").replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"<\w>(.*?)</\w>", r"\1", text)
    text = re.sub(r"[\ue000-\uf8ff]", "", text)

    # 4. Пометки листов {л._1} и пропуски (...)
    text = re.sub(r"\{.*?\}", "", text)
    text = re.sub(r"[(\[<]\s*\.\.\.\s*[)\]>]", "", text)
    text = re.sub(r"[\[\]()<>'ʼ]", "", text)

    # 5. Меняем подчеркивания и слеши на пробелы (расклеиваем слова И_не -> И не)
    text = text.replace("_", " ").replace("|", " ")

    # 6. АГРЕССИВНАЯ ЗАЧИСТКА: Удаляем все арабские цифры и остатки латиницы
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[a-zA-Z]", "", text)

    # 7. Удаляем остатки странной пунктуации
    text = re.sub(r"[@#$%^&*=+<>\/\\~`{}\[\]]", "", text)

    # 8. Умная очистка диакритики (оставляем титла!)
    nfd_form = unicodedata.normalize("NFD", text)
    clean_chars = [
        c for c in nfd_form if unicodedata.category(c) != "Mn" or ord(c) in TITLO_RANGE
    ]
    clean_text = unicodedata.normalize("NFC", "".join(clean_chars))
    text = re.sub(r"[\u0300-\u036f]", "", clean_text)

    # 9. Переводим ВСЁ в нижний регистр (решает проблему скачущего капса: сЛОВо -> слово)
    text = text.lower()

    # 10. Схлопываем пробелы и убираем мусор в начале
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[.\s\-\–\—\:]+", "", text)

    if not text:
        return ""

    # 11. Делаем первую букву заглавной
    text = text[0].upper() + text[1:]
    return text


def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Ошибка: Папка {INPUT_FOLDER} не найдена.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    all_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")])
    print(f"📂 Начинаем очистку {len(all_files)} файлов...\n")

    total_lines = 0
    total_tokens = 0

    # Список для сбора статистики (пойдет в CSV)
    csv_data = [["Filename", "Lines", "Tokens"]]

    for filename in all_files:
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        try:
            with open(
                input_path, "r", encoding="utf-8", errors="ignore"
            ) as infile, open(output_path, "w", encoding="utf-8") as outfile:

                content = infile.read()
                sentences = re.split(r"(?<=[.!?|;:\n])\s+", content)

                file_lines_count = 0
                file_tokens_count = 0

                for sent in sentences:
                    clean_line = clean_diacu_text(sent)
                    for chunk in split_long_line(clean_line):
                        # Проверяем на минимальную длину, наличие кириллицы и минимум 2 слова
                        if (
                            len(chunk) > 15
                            and len(chunk.split()) > 1
                            and re.search(r"[а-яА-ЯёЁѣѢіІѵѴѫѸѡѠѕЅ]", chunk)
                        ):
                            outfile.write(f"{chunk}\n")
                            file_lines_count += 1

                            # Подсчет токенов (слова + знаки препинания)
                            tokens_in_chunk = len(re.findall(r"\w+|[^\w\s]", chunk))
                            file_tokens_count += tokens_in_chunk

                # Выводим стату в консоль и сохраняем в список для CSV
                if file_lines_count > 0:
                    print(
                        f"✅ {filename[:40]:<42} | Строк: {file_lines_count:<6} | Токенов: {file_tokens_count}"
                    )
                    csv_data.append([filename, file_lines_count, file_tokens_count])

                total_lines += file_lines_count
                total_tokens += file_tokens_count

        except Exception as e:
            print(f"⚠️ Ошибка {filename}: {e}")

    # Сохраняем статистику в CSV
    with open(CSV_OUTPUT_FILE, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)

    print("\n" + "=" * 50)
    print(f"🔥 ГОТОВО! Очищено файлов: {len(all_files)}. Лежат в '{OUTPUT_FOLDER}'")
    print(f"📊 Статистика сохранена в файл: {CSV_OUTPUT_FILE}")
    print(f"Всего подготовлено строк: {total_lines}")
    print(f"Всего токенов (TOROT style): {total_tokens}")
    print("=" * 50)


if __name__ == "__main__":
    main()
