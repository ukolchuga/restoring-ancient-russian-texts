import os
import pandas as pd
import re


def clean_legal_text(text):
    """
    Функция для очистки юридических текстов от нумерации статей и глав.
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 1. Убираем номера статей в начале строки (например "1. Будет кто..." -> "Будет кто...")
        # Ищет: Начало строки (^), цифры (\d+), точку (\.), пробелы (\s*)
        line = re.sub(r"^\d+\.\s*", "", line)

        # 2. Убираем слова "Глава N", если они стоят отдельно
        # Ищет: Слово Глава, пробел, цифры или римские цифры (IVX), точку, пробел
        line = re.sub(r"^глава\s+[\divx]+\.?\s*", "", line, flags=re.IGNORECASE)

        # 3. Убираем конструкции вида "Статья 5."
        line = re.sub(r"^статья\s+\d+\.?\s*", "", line, flags=re.IGNORECASE)

        # Сохраняем строку, только если в ней осталось больше 3 символов (чтобы убрать мусор)
        if len(line) > 3:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def prepare_final_dataset():
    # Файлы источников
    file_torot = "torot_corpus_final.txt"
    file_pushkin = "pushkinskij_full.txt"
    file_bible = "bible_full_clean.txt"

    # Юридические тексты (которые надо чистить)
    file_sudebnic_1497 = "sudebnic_1497.txt"
    file_sudebnic_1550 = (
        "sudebnic_1550.txt"  # Если он есть (обычно Судебник 1550, но вдруг у вас 1450)
    )
    file_sudebnoe_izlozhenie = "sobornoe_izlozhenie.txt"

    file_csv_gramoty = "gramoty_train_fixed.csv"
    csv_text_column = "text"

    output_file = "final_dataset_ready.txt"

    print(">>> НАЧИНАЕМ ОБЪЕДИНЕНИЕ ДАННЫХ (С ОЧИСТКОЙ ЗАКОНОВ) <<<")

    total_lines = 0

    with open(output_file, "w", encoding="utf-8") as outfile:

        # 1. Читаем TOROT (TXT)
        if os.path.exists(file_torot):
            print(f"1. Читаем {file_torot}...", end=" ")
            with open(file_torot, "r", encoding="utf-8") as f:
                text = f.read()
                outfile.write(text + "\n\n")
                lines = len(text.splitlines())
                total_lines += lines
            print(f"OK ({lines} строк)")
        else:
            print(f"\n⚠️ Файл {file_torot} не найден.")

        # 2. Читаем Пушкинский дом (TXT)
        if os.path.exists(file_pushkin):
            print(f"2. Читаем {file_pushkin}...", end=" ")
            with open(file_pushkin, "r", encoding="utf-8") as f:
                text = f.read()
                outfile.write(text + "\n\n")
                lines = len(text.splitlines())
                total_lines += lines
            print(f"OK ({lines} строк)")
        else:
            print(f"\n⚠️ Файл {file_pushkin} не найден.")

        # 3. Читаем Библию
        if os.path.exists(file_bible):
            print(f"3. Читаем {file_bible}...", end=" ")
            with open(file_bible, "r", encoding="utf-8") as f:
                text = f.read()
                outfile.write(text + "\n\n")
                lines = len(text.splitlines())
                total_lines += lines
            print(f"OK ({lines} строк)")
        else:
            print(f"\n⚠️ Файл {file_bible} не найден.")

        # 4. Читаем Грамоты (CSV)
        if os.path.exists(file_csv_gramoty):
            print(f"4. Обрабатываем CSV {file_csv_gramoty}...", end=" ")
            try:
                df = pd.read_csv(file_csv_gramoty)

                if csv_text_column in df.columns:
                    gramoty_texts = df[csv_text_column].dropna().astype(str).tolist()

                    outfile.write("\n\n--- БЕРЕСТЯНЫЕ ГРАМОТЫ ---\n\n")
                    for g_text in gramoty_texts:
                        clean_text = g_text.strip()
                        if len(clean_text) > 1:
                            outfile.write(clean_text + "\n")

                    count = len(gramoty_texts)
                    total_lines += count
                    print(f"OK (добавлено {count} грамот)")
                else:
                    print(f"\n❌ ОШИБКА: В CSV нет колонки '{csv_text_column}'.")
            except Exception as e:
                print(f"\n❌ Ошибка чтения CSV: {e}")
        else:
            print(f"\n⚠️ Файл {file_csv_gramoty} не найден.")

        # 5. Читаем Судебник 1497 (С ОЧИСТКОЙ)
        if os.path.exists(file_sudebnic_1497):
            print(f"5. Читаем {file_sudebnic_1497}...", end=" ")
            with open(file_sudebnic_1497, "r", encoding="utf-8") as f:
                text = f.read()
                # Сначала чистим от номеров
                text = clean_legal_text(text)
                # Потом в нижний регистр (как вы хотели)
                text = text.lower()

                outfile.write(text + "\n\n")
                lines = len(text.splitlines())
                total_lines += lines
            print(f"OK ({lines} строк)")
        else:
            print(f"\n⚠️ Файл {file_sudebnic_1497} не найден.")

        # 6. Читаем Судебник 1450/1550 (С ОЧИСТКОЙ)
        if os.path.exists(file_sudebnic_1550):
            print(f"6. Читаем {file_sudebnic_1550}...", end=" ")
            with open(file_sudebnic_1550, "r", encoding="utf-8") as f:
                text = f.read()
                text = clean_legal_text(text)
                text = text.lower()

                outfile.write(text + "\n\n")
                lines = len(text.splitlines())
                total_lines += lines
            print(f"OK ({lines} строк)")
        else:
            print(f"\n⚠️ Файл {file_sudebnic_1550} не найден.")

        # 7. Читаем Соборное Уложение (С ОЧИСТКОЙ)
        if os.path.exists(file_sudebnoe_izlozhenie):
            print(f"7. Читаем {file_sudebnoe_izlozhenie}...", end=" ")
            with open(file_sudebnoe_izlozhenie, "r", encoding="utf-8") as f:
                text = f.read()
                text = clean_legal_text(text)
                text = text.lower()

                outfile.write(text + "\n\n")
                lines = len(text.splitlines())
                total_lines += lines
            print(f"OK ({lines} строк)")
        else:
            print(f"\n⚠️ Файл {file_sudebnoe_izlozhenie} не найден.")

    print("-" * 30)
    print(f"ГОТОВО! Итоговый файл: {output_file}")
    print(f"Всего строк для обучения: {total_lines}")


if __name__ == "__main__":
    prepare_final_dataset()
