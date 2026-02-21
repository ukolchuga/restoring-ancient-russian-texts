import os

# ПАРАМЕТРЫ
INPUT_FOLDER = "diacu_cleaned_texts"  # Папка, где лежат очищенные файлы DIACU
OUTPUT_FILE = "diacu_full.txt"  # Итоговый файл


def main():
    if not os.path.exists(INPUT_FOLDER):
        print(
            f"❌ Ошибка: Папка '{INPUT_FOLDER}' не найдена. Убедись, что скрипт очистки отработал."
        )
        return

    # Берем все txt файлы и сортируем их по алфавиту
    all_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")])
    print(f"📂 Найдено файлов для склейки: {len(all_files)}...\n")

    total_lines = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for filename in all_files:
            filepath = os.path.join(INPUT_FOLDER, filename)

            try:
                with open(filepath, "r", encoding="utf-8") as infile:
                    content = infile.read().strip()

                    if content:
                        # Записываем контент файла и добавляем перенос строки,
                        # чтобы тексты из разных файлов не слиплись
                        outfile.write(content + "\n")

                        # Считаем количество строк для финальной статистики
                        lines_count = len(content.split("\n"))
                        total_lines += lines_count

            except Exception as e:
                print(f"⚠️ Ошибка при чтении {filename}: {e}")

    print(f"🔥 ГОТОВО! Все файлы успешно объединены.")
    print(f"📁 Итоговый файл: {OUTPUT_FILE}")
    print(f"📈 Всего строк: {total_lines}")


if __name__ == "__main__":
    main()
