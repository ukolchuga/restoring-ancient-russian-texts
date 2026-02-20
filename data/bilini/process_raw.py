import re
import os

# ПАРАМЕТРЫ
INPUT_FOLDER = "raw_texts"
OUTPUT_FILE = "ultimate_ancient_rus_corpus.txt"


def clean_ancient_travelogue(content):
    # 1. Удаляем сноски [1], [25]
    content = re.sub(r"\[\d+\]", "", content)

    # 2. Удаляем пустые скобки () и <>
    content = re.sub(r"\(\s*\)|<\s*>", "", content)
    # Удаляем текст в скобках (обычно современные пояснения)
    content = re.sub(r"\(.*?\)", "", content)

    # 3. Чистим номера строк (цифры в начале строк текста)
    # Это уберет "160 В славном..." -> "В славном..."
    content = re.sub(r"^\s*\d+\s*", "", content, flags=re.M)

    # 4. Технический мусор
    content = re.sub(r"[\*\+\=\xa0]", " ", content)

    # 5. РАЗБИВКА: сначала делим на физические строки, чтобы не склеить всю былину в ком
    lines = content.splitlines()
    clean_sentences = []

    for line in lines:
        # Убираем лишние пробелы
        line = line.strip()

        # Если строка целиком капсом — переводим в нормальный вид (как предложение)
        if line.isupper():
            line = line.capitalize()

        # Фильтр: длина > 20 (чтобы отсечь мелкие обрывки)
        # и наличие кириллицы
        if len(line) > 20 and re.search(r"[а-яА-ЯёЁѣѢіІѵѴѫѸ]", line):
            # Финальная чистка двойных пробелов внутри
            line = re.sub(r"\s+", " ", line)
            clean_sentences.append(line)

    return clean_sentences


def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Ошибка: Папка {INPUT_FOLDER} не найдена.")
        return

    all_files_lines = []
    files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")])

    print(f"🚀 Начинаем обработку {len(files)} файлов...\n")
    print(f"{'Название файла':<40} | {'Строк':<10}")
    print("-" * 55)

    total_sum = 0

    for filename in files:
        path = os.path.join(INPUT_FOLDER, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            sentences = clean_ancient_travelogue(content)
            file_count = len(sentences)

            for s in sentences:
                all_files_lines.append(s)

            print(f"{filename:<40} | {file_count:<10}")
            total_sum += file_count

        except Exception as e:
            print(f"⚠️ Ошибка в {filename}: {e}")

    # Сохраняем итоговый корпус
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(all_files_lines))

    print("-" * 55)
    print(f"✅ Сборка завершена. Всего строк в корпусе: {total_sum}")


if __name__ == "__main__":
    main()
