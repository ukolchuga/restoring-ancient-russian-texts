import re
import os
from tqdm import tqdm

INPUT_FILE = "final_ancient_rus_dataset.txt"  # Твой собранный мега-корпус
OUTPUT_FILE = "ancient_rus_ready_for_bert.txt"  # Финальный, отполированный файл


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

    # 2. УДАЛЯЕМ НЕВИДИМЫЙ МУСОР
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"[\ue000-\uf8ff]", "", text)  # Private Use Area

    # 2.5. УДАЛЯЕМ ВСЕ ВИДЫ КАВЫЧЕК (Новый шаг!)
    # Убираем елочки, лапки, прямые двойные и одинарные кавычки
    text = re.sub(r'["\'«»„“]', "", text)

    # 3. УДАЛЯЕМ УДАРЕНИЯ
    text = re.sub(r"[\u0300-\u036f]", "", text)

    # 4. БЕЗОПАСНАЯ ЗАМЕНА ЛАТИНИЦЫ НА КИРИЛЛИЦУ
    text = text.replace("[UNK]", "___999999___")

    replacements = {
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
        "X": "Х",
        "x": "х",
    }
    for lat, cyr in replacements.items():
        text = text.replace(lat, cyr)

    text = text.replace("___999999___", "[UNK]")

    # 5. РАЗВЕРТЫВАНИЕ ПОТЕРЯННЫХ ТИТЛ
    TITLO = r"[\u0483-\u0489]"
    abbrev_map = {
        rf"\bбг\b(?!{TITLO})": "богъ",
        rf"\bгд\b(?!{TITLO})": "господь",
        rf"\bсн\b(?!{TITLO})": "сынъ",
        rf"\bхс\b(?!{TITLO})": "христосъ",
        rf"\bгн\b(?!{TITLO})": "господинъ",
    }
    for pattern, repl in abbrev_map.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # 6. ФИНАЛЬНАЯ ЧИСТКА ПРОБЕЛОВ
    text = re.sub(r"\s+", " ", text).strip()

    # Собираем обратно
    if tag:
        return f"{tag} {text}"
    return text


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Ошибка: Файл {INPUT_FILE} не найден!")
        return

    print(f"🧹 Начинаем финальную полировку Мега-Корпуса...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, open(
        OUTPUT_FILE, "w", encoding="utf-8"
    ) as f_out:

        lines = f_in.readlines()
        cleaned_count = 0

        for line in tqdm(lines, desc="Очистка строк"):
            cleaned = safe_clean_text(line)
            if len(cleaned) > 5:
                f_out.write(cleaned + "\n")
                cleaned_count += 1

    print("\n" + "=" * 50)
    print("✨ ПОЛИРОВКА УСПЕШНО ЗАВЕРШЕНА ✨")
    print(f"Финальный чистейший файл: {OUTPUT_FILE}")
    print(f"Сохранено строк: {cleaned_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
