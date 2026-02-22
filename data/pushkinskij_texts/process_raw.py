import os
import re
import csv
from tqdm import tqdm

# ПАРАМЕТРЫ
INPUT_FOLDER = "raw_texts"  # Папка с исходными (сырыми) файлами
OUTPUT_FOLDER = (
    "clean_texts"  # Папка, куда сохранятся очищенные файлы (структура сохранится)
)
CSV_OUTPUT_FILE = "pushkinskij_metadata.csv"  # Файл для Google Таблиц

# Список папок, которые скрипт будет искать
FOLDERS = ["CHURCH", "LIT", "DAILY", "LEGAL", "SCIENCE", "EPIC"]


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


def clean_ancient_text(text):
    if not text:
        return ""

    text = re.sub(r"[(\[<]\s*\.\.\.\s*[)\]>]", "", text)
    text = text.replace("<", "").replace(">", "")
    text = re.sub(r"\(\s*\)|\[\s*\]|\{\s*\}", "", text)
    text = re.sub(r"\b([А-ЯЁѢѤІѴѸѲѺ]{2,})\b", lambda m: m.group(1).lower(), text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(\d+\)", "", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)
    text = re.sub(r"^[.\s\-\–\—\:]+", "", text)
    text = normalize_caps(text)
    text = re.sub(r"^\d+\s+", "", text, flags=re.M)
    text = text.replace("\t", " ").replace("...", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[()\[\]{}.\s\-\–\—\:]+", "", text)

    if not text:
        return ""

    text = text[0].upper() + text[1:]
    return text


def normalize_caps(text):
    if text.isupper():
        return text.capitalize()
    return text


def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Ошибка: Главная папка {INPUT_FOLDER} не найдена.")
        return

    # Создаем базовую выходную папку
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    csv_data = [["Folder", "Filename", "Lines", "Tokens"]]
    total_lines = 0
    total_words = 0
    files_processed = 0

    # Проходим по нужным папкам
    for folder_name in FOLDERS:
        input_folder_path = os.path.join(INPUT_FOLDER, folder_name)
        output_folder_path = os.path.join(OUTPUT_FOLDER, folder_name)

        if not os.path.exists(input_folder_path):
            print(
                f"⚠️ Папка {folder_name} не найдена внутри {INPUT_FOLDER}, пропускаем."
            )
            continue

        all_files = sorted(
            [f for f in os.listdir(input_folder_path) if f.endswith(".txt")]
        )
        if not all_files:
            continue

        # Создаем аналогичную папку в clean_texts
        os.makedirs(output_folder_path, exist_ok=True)
        print(f"\n📁 ОБРАБОТКА ПАПКИ: {folder_name}")

        for filename in all_files:
            in_filepath = os.path.join(input_folder_path, filename)
            out_filepath = os.path.join(output_folder_path, filename)
            files_processed += 1

            try:
                with open(in_filepath, "r", encoding="utf-8") as infile:
                    raw_content = infile.read()

                sentences = re.split(r"(?<=[.!?|;:])\s+", raw_content)

                file_lines_count = 0
                file_words_count = 0

                # Открываем индивидуальный выходной файл на запись
                with open(out_filepath, "w", encoding="utf-8") as outfile:
                    for sent in sentences:
                        clean_line = clean_ancient_text(sent)
                        for chunk in split_long_line(clean_line):
                            if len(chunk) > 15:
                                if not re.search(r"[a-zA-Z]", chunk):
                                    outfile.write(f"{chunk}\n")
                                    file_lines_count += 1

                                    words_in_chunk = len(
                                        re.findall(r"\w+|[^\w\s]", chunk)
                                    )
                                    file_words_count += words_in_chunk

                # Если после очистки файл оказался пустым (например, сплошная латиница), удаляем его
                if file_lines_count == 0:
                    os.remove(out_filepath)
                else:
                    print(
                        f"✅ {filename[:40]:<42} | Строк: {file_lines_count:<5} | Токенов: {file_words_count}"
                    )
                    csv_data.append(
                        [folder_name, filename, file_lines_count, file_words_count]
                    )

                total_lines += file_lines_count
                total_words += file_words_count

            except Exception as e:
                print(f"⚠️ Ошибка при чтении {filename}: {e}")

    with open(CSV_OUTPUT_FILE, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)

    print("\n" + "=" * 50)
    print(f"🔥 ГОТОВО! Обработано файлов: {files_processed}")
    print(f"📂 Очищенные файлы сохранены в папке: {OUTPUT_FOLDER}")
    print(f"📊 Метадата по папкам сохранена в: {CSV_OUTPUT_FILE}")
    print(f"Всего строк: {total_lines}")
    print(f"Всего токенов: {total_words}")
    print("=" * 50)


if __name__ == "__main__":
    main()
