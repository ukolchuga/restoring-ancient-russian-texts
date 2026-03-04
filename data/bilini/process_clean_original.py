import os
import re
import pandas as pd


def build_dataset_and_metadata(input_folder, output_txt, output_csv, gap_tag="[GAP]"):
    """
    Объединяет txt-файлы, заменяет пропуски на единый тег,
    удаляет ВСЕ пустые строки, считает токены и генерирует CSV.
    """
    if not os.path.exists(input_folder):
        print(f"Ошибка: Папка '{input_folder}' не найдена!")
        return

    combined_dataset = []
    metadata = []

    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_folder, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            # 1. Замена пропусков на тег
            text = re.sub(r"(?:\s*\.{2,}\s*)+", f" {gap_tag} ", text)

            # 2. УДАЛЕНИЕ ПУСТЫХ СТРОК (Самый надежный метод)
            # Разбиваем весь текст построчно.
            # if line.strip() гарантирует, что мы берем только те строки, где есть текст.
            valid_lines = [line.strip() for line in text.splitlines() if line.strip()]

            # Склеиваем строки обратно через одинарный перенос
            clean_text = "\n".join(valid_lines)

            # Убираем случайные двойные пробелы внутри самих строк
            clean_text = re.sub(r"[ \t]+", " ", clean_text)

            # Если после чистки в файле вообще остался текст, добавляем его
            if clean_text:
                combined_dataset.append(clean_text)

                # 3. Подсчет токенов
                tokens = re.findall(r"\w+|[^\w\s]", clean_text)

                metadata.append(
                    {
                        "Имя_файла": filename,
                        "Количество_токенов": len(tokens),
                        "Размер_символов": len(clean_text),
                    }
                )

    # --- СОХРАНЕНИЕ ДАННЫХ ---

    # Записываем объединенный датасет.
    # Используем '\n' вместо '\n\n', чтобы между файлами тоже не было пустых строк.
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_dataset))

    # Сохраняем статистику в CSV
    if metadata:
        df = pd.DataFrame(metadata)
        total_tokens = df["Количество_токенов"].sum()
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        print("-" * 50)
        print("Генерация плотного корпуса завершена!")
        print(f"Обработано файлов: {len(metadata)}")
        print(f"Всего токенов: {total_tokens}")
        print(f"Корпус сохранен в: {output_txt}")
        print(f"CSV метадата: {output_csv}")
        print("-" * 50)
    else:
        print("Не найдено текста для сохранения!")


if __name__ == "__main__":
    INPUT_FOLDER = "clean_original"
    OUTPUT_TXT = "dobrynya_byliny_corpus.txt"
    OUTPUT_CSV = "dobrynya_metadata.csv"
    GAP_TAG = "[GAP]"

    build_dataset_and_metadata(INPUT_FOLDER, OUTPUT_TXT, OUTPUT_CSV, gap_tag=GAP_TAG)
