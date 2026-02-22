import os
import glob
import re
from tqdm import tqdm

DATA_DIR = "./torot_data"
OUTPUT_DIR = "torot_cleaned_texts"


def clean_torot_sentence(text):
    # 1. Убираем все виды скобок (квадратные, круглые, фигурные, угловые)
    text = re.sub(r"[\[\]\(\)\{\}\<\>]", "", text)

    # 2. Убираем точки, которые разрывают слова (ИМѢ.ЕТЪ -> ИМѢЕТЪ)
    text = text.replace(".", "")

    # 3. Убираем КАПС и делаем Sentence case (Первая буква заглавная, остальные строчные)
    # Пример: "КНИГА ГЛАГОЛЕМАЯ" -> "Книга глаголемая"
    text = text.capitalize()

    # 4. Убираем лишние пробелы (если они появились после удаления символов)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_conll_file(filepath):
    sentences = []
    current_sentence = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Пустая строка означает конец предложения
            if not line:
                if current_sentence:
                    text = " ".join(current_sentence)
                    # === ПРОПУСКАЕМ ТЕКСТ ЧЕРЕЗ ЧИСТИЛЬЩИК ===
                    clean_text = clean_torot_sentence(text)
                    if len(clean_text) > 2:
                        sentences.append(clean_text)
                current_sentence = []
                continue

            # Пропускаем комментарии
            if line.startswith("#"):
                continue

            # Берем само слово (2-я колонка в CoNLL)
            parts = line.split("\t")
            if len(parts) > 1:
                word = parts[1]

                if word != "_" and word.strip():
                    current_sentence.append(word)

    # Захватываем последнее предложение в файле
    if current_sentence:
        text = " ".join(current_sentence)
        clean_text = clean_torot_sentence(text)
        if len(clean_text) > 2:
            sentences.append(clean_text)

    return sentences


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_files = glob.glob(os.path.join(DATA_DIR, "*.conll"))

    print(f"📂 Найдено файлов TOROT для парсинга: {len(all_files)}")
    total_valid_sentences = 0

    for filepath in tqdm(all_files, desc="Парсинг и очистка"):
        try:
            sentences = parse_conll_file(filepath)

            if not sentences:
                continue

            # Получаем чистое имя файла (например: "domo")
            base_filename = os.path.splitext(os.path.basename(filepath))[0]
            output_filepath = os.path.join(OUTPUT_DIR, f"{base_filename}.txt")

            with open(output_filepath, "w", encoding="utf-8") as f_out:
                for sent in sentences:
                    f_out.write(sent + "\n")

            total_valid_sentences += len(sentences)

        except Exception as e:
            print(f"❌ Ошибка при обработке {filepath}: {e}")

    print("\n" + "=" * 50)
    print("✨ ПАРСИНГ И ОЧИСТКА УСПЕШНО ЗАВЕРШЕНЫ ✨")
    print(f"Папка с чистыми результатами: {OUTPUT_DIR}/")
    print(f"Всего файлов создано: {len(all_files)}")
    print(f"Всего предложений извлечено: {total_valid_sentences}")
    print("=" * 50)


if __name__ == "__main__":
    main()
