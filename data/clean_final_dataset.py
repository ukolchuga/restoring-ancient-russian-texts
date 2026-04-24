import os
import re
import unicodedata

from tqdm import tqdm

INPUT_FILE = "final_ancient_rus_dataset.txt"
OUTPUT_FILE = "ancient_rus_ready_for_bert.txt"


def safe_clean_text(line):
    line = line.strip()
    if not line:
        return ""

    # 1. ОТДЕЛЯЕМ ТЕГ ОТ ТЕКСТА
    match = re.match(r"^(\[CTX_[A-Z_]+\])\s+(.*)", line)
    if match:
        tag = match.group(1)
        text = match.group(2)
    else:
        tag = ""
        text = line

    # 2. NFC НОРМАЛИЗАЦИЯ (Критично для правильных титл)
    text = unicodedata.normalize("NFC", text)

    # 3. УДАЛЯЕМ НЕВИДИМЫЙ МУСОР И КАВЫЧКИ
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"[\ue000-\uf8ff]", "", text)  # Private Use Area
    text = re.sub(r'["\'«»„“]', "", text)

    # 4. УДАЛЯЕМ СОВРЕМЕННЫЕ УДАРЕНИЯ (Диапазон 0300-036F)
    # Знак титла (0483-0489) сюда не попадает, он в безопасности!
    text = re.sub(r"[\u0300-\u036f]", "", text)

    # 5. БЕЗОПАСНАЯ ЗАМЕНА ЛАТИНИЦЫ НА КИРИЛЛИЦУ
    # Прячем наш GAP, чтобы латинские G, A, P не заменились
    text = text.replace("[GAP]", "___999999___")

    replacements = {
        "A": "А", "a": "а",
        "B": "В", "b": "в",
        "E": "Е", "e": "е",
        "K": "К", "k": "к",
        "M": "М", "m": "м",
        "H": "Н", "n": "н",
        "O": "О", "o": "о",
        "P": "Р", "p": "р",
        "C": "С", "c": "с",
        "T": "Т", "t": "т",
        "y": "у", "x": "х", "X": "Х",
        "i": "і", "I": "І",
    }
    for lat, cyr in replacements.items():
        text = text.replace(lat, cyr)

    text = text.replace("___999999___", "[GAP]")

    # 6. ГЛОБАЛЬНАЯ ИЗОЛЯЦИЯ ДРЕВНЕЙ ПУНКТУАЦИИ
    text = re.sub(r"([·:⁘])", r" \1 ", text)

    # 7. ФИНАЛЬНАЯ ЧИСТКА ПРОБЕЛОВ
    text = re.sub(r"(\s*\[GAP\]\s*)+", " [GAP] ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Собираем обратно
    if tag:
        return f"{tag} {text}"
    return text


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Ошибка: Файл {INPUT_FILE} не найден!")
        return

    print("🧹 Начинаем финальную полировку Мега-Корпуса...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()

    cleaned_count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for line in tqdm(lines, desc="Очистка строк"):
            cleaned = safe_clean_text(line)
            # Проверяем наличие букв
            letters = re.findall(r"[а-яА-ЯёЁ\u0400-\u052F\uA640-\uA69F]", cleaned)
            if len(letters) >= 3:
                f_out.write(cleaned + "\n")
                cleaned_count += 1

    print("\n" + "=" * 50)
    print("✨ ПОЛИРОВКА УСПЕШНО ЗАВЕРШЕНА ✨")
    print(f"Финальный чистейший файл: {OUTPUT_FILE}")
    print(f"Сохранено строк: {cleaned_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
