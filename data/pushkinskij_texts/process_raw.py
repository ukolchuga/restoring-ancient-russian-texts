import os
import re

# ПАРАМЕТРЫ
INPUT_FOLDER = "raw_texts"  # Папка с исходными txt
OUTPUT_FILE = "pushkinskij_full.txt"  # Куда сохраняем результат


def split_long_line(text, max_len=500):
    """Если строка все еще слишком длинная, режем ее по пробелам."""
    if len(text) <= max_len:
        return [text]

    parts = []
    while len(text) > max_len:
        # Ищем последний пробел в пределах max_len
        split_idx = text.rfind(" ", 0, max_len)
        if split_idx == -1:  # Если пробелов нет, режем жестко
            split_idx = max_len

        parts.append(text[:split_idx].strip())
        text = text[split_idx:].strip()

    if text:
        parts.append(text)
    return parts


def clean_ancient_text(text):
    if not text:
        return ""

    # 0. Убираем редакторские пропуски: (...), [...], <...>
    text = re.sub(r"[(\[<]\s*\.\.\.\s*[)\]>]", "", text)

    # 1. Восстанавливаем реконструированный текст из угловых скобок
    text = text.replace("<", "").replace(">", "")

    # 2. Удаляем ПУСТЫЕ скобки (любого вида)
    text = re.sub(r"\(\s*\)|\[\s*\]|\{\s*\}", "", text)

    # 3. Переводим слова КАПСОМ в нижний регистр (если 2 и более букв)
    text = re.sub(r"\b([А-ЯЁѢѤІѴѸѲѺ]{2,})\b", lambda m: m.group(1).lower(), text)

    # 4. Убираем сноски [1], (1)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(\d+\)", "", text)

    # 5. Убираем ударения (диакритику) - КРИТИЧНО для ПД
    text = re.sub(r"[\u0300-\u036f]", "", text)

    # 6. Убираем мусор в начале строки
    text = re.sub(r"^[.\s\-\–\—\:]+", "", text)

    # 7. Нормализуем капс (переводим в нижний регистр, если вся строка заглавными)
    text = normalize_caps(text)

    # 8. Убираем номера строк и страниц в начале
    text = re.sub(r"^\d+\s+", "", text, flags=re.M)

    # 9. Очистка пунктуации (табы и оставшиеся одиночные многоточия)
    text = text.replace("\t", " ").replace("...", "")

    # 10. Схлопываем множественные пробелы
    text = re.sub(r"\s+", " ", text).strip()

    # 11. Финальная зачистка начала строки (убираем мусор, который мог остаться после шагов 1-2)
    text = re.sub(r"^[()\[\]{}.\s\-\–\—\:]+", "", text)

    if not text:
        return ""

    # 12. Делаем первую букву заглавной
    text = text[0].upper() + text[1:]

    return text


def normalize_caps(text):
    if text.isupper():
        # Если вся строка капсом, делаем ее как предложение
        return text.capitalize()
    return text


def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Ошибка: Папка {INPUT_FOLDER} не найдена.")
        return

    all_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]
    print(f"📂 Найдено файлов: {len(all_files)}")

    total_lines = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for filename in all_files:
            filepath = os.path.join(INPUT_FOLDER, filename)

            try:
                with open(filepath, "r", encoding="utf-8") as infile:
                    raw_content = infile.read()

                # Разбиваем на предложения (учитываем сокращения)
                sentences = re.split(r"(?<=[.!?|;:])\s+", raw_content)

                file_lines_count = 0
                for sent in sentences:
                    clean_line = clean_ancient_text(sent)
                    for chunk in split_long_line(clean_line):
                        if len(chunk) > 15:
                            if not re.search(r"[a-zA-Z]", chunk):
                                outfile.write(f"{chunk}\n")
                                file_lines_count += 1

                print(f"✅ {filename}: сохранено {file_lines_count} строк.")
                total_lines += file_lines_count

            except Exception as e:
                print(f"⚠️ Ошибка при чтении {filename}: {e}")

    print(f"\n🔥 ГОТОВО! Итоговый файл: {OUTPUT_FILE}")
    print(f"Всего строк для обучения: {total_lines}")


if __name__ == "__main__":
    main()
