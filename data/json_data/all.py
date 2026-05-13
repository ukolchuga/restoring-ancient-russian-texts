import json
import os
import re
import unicodedata
import csv
from tqdm import tqdm

# ПАРАМЕТРЫ
JSON_PATH = "DIACU_1.0.json"
OUTPUT_FOLDER = "diacu_cleaned_texts"  # Папка для чистых текстовых файлов
CSV_OUTPUT_FILE = "diacu_metadata.csv"  # Файл для Google Таблиц

TITLO_RANGE = range(0x0483, 0x0488)
CYR_NUMERALS = "авгдєѕзиѳіклмнѯопрстуфхѱѡцчшщъыьѣюѧѩѫѷѵ"
NUM_PATTERN = re.compile(rf"([:+·])([{CYR_NUMERALS}]+҃)\1")

PUNCT_MAP = {
    "†": "+",
    "×": "+",
    "*": "+",
    "⁘": ":",
    "⁙": ":",
    "⁞": ":",
    "¦": ":",
    "∙": "·",
    ".": "·",
    "҂": "·",
    "\uf13f": "·",
}
# Обновленный словарь: спасаем опечатки оцифровщиков
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


def sanitize_filename(name):
    """Твоя функция для создания безопасных имен файлов."""
    name = re.sub(r"[^\w\-_]", "_", name)
    return re.sub(r"_{2,}", "_", name)[:50].strip("_")


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


def _protect_numerals(text: str):
    protected_nums = {}

    def repl(m):
        key = f"PNUM{len(protected_nums)}PNUM"
        protected_nums[key] = m.group(0)
        return key

    return NUM_PATTERN.sub(repl, text), protected_nums


def _unprotect_numerals(text: str, protected_nums: dict[str, str]) -> str:
    for key, value in protected_nums.items():
        text = text.replace(key, value)
    return text

def clean_diacu_text(text):
    if not text:
        return ""

    # 1. Спасаем слова (латиница -> кириллица)
    for lat, cyr in REPLACEMENTS.items():
        text = text.replace(lat, cyr)

    # 2. Восстанавливаем древнерусский знак тысячи (҂)
    text = text.replace("$", "҂").replace("@", "҂")

    # 3. Очистка от технического мусора
    text = text.replace("\x00", "").replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"<\w>(.*?)</\w>", r"\1", text)
    text = re.sub(r"[\ue000-\uf8ff]", "", text)

    # 4. Пометки листов {л._1} и пропуски (...)
    text = re.sub(r"\{.*?\}", "", text)
    text = re.sub(r"[(\[<]\s*\.\.\.\s*[)\]>]", "", text)
    text = re.sub(r"[\[\]()<>'ʼ]", "", text)

    # 5. Меняем подчеркивания и слеши на пробелы
    text = text.replace("_", " ").replace("|", " ")

    # 6. Агрессивно удаляем цифры и остаточную латиницу
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[a-zA-Z]", "", text)

    # 7. Нормализуем пунктуацию:
    #    - часть знаков сводим к + : ·
    #    - всю остальную пунктуацию превращаем в пробелы
    cleaned_chars = []
    for ch in text:
        if ch in PUNCT_MAP:
            cleaned_chars.append(PUNCT_MAP[ch])
            continue

        if ch in {"+", ":", "·"}:
            cleaned_chars.append(ch)
            continue

        cat = unicodedata.category(ch)
        if cat[0] in {"L", "N", "M"} or ch.isspace():
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")

    text = "".join(cleaned_chars)

    # 8. Защищаем древнерусские числительные
    text, protected_nums = _protect_numerals(text)

    # 9. Отделяем всю пунктуацию пробелами от всего остального
    #    (числительные уже защищены, так что их это не затронет)
    text = re.sub(r"\s*([:+·])\s*", r" \1 ", text)

    # 10. Возвращаем числительные обратно
    text = _unprotect_numerals(text, protected_nums)

    # 11. Умная очистка диакритики:
    #     оставляем только титло, а все его варианты приводим к одному символу ҃
    normalized = []
    for ch in unicodedata.normalize("NFC", text):
        if ord(ch) in TITLO_RANGE:
            normalized.append("҃")
        else:
            normalized.append(ch)
    text = "".join(normalized)

    # 12. Ещё раз удалим лишние combining marks, но титло сохраним
    nfd_form = unicodedata.normalize("NFD", text)
    clean_chars = [
        c for c in nfd_form
        if unicodedata.category(c) != "Mn" or c == "҃"
    ]
    text = unicodedata.normalize("NFC", "".join(clean_chars))

    # 13. Переводим всё в нижний регистр
    text = text.lower()

    # 14. Схлопываем пробелы
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # 15. Подчищаем мусор в начале строки
    text = re.sub(r"^[.\s\-\–\—\:]+", "", text)

    if not text:
        return ""

    # text = text[0].upper() + text[1:]
    return text

def main():
    if not os.path.exists(JSON_PATH):
        print(f"❌ Ошибка: Файл {JSON_PATH} не найден.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"⏳ Чтение {JSON_PATH}...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_docs = data.get("Documents", [])
    print(f"📂 Найдено документов: {len(raw_docs)}\n")

    # Заголовки для CSV
    csv_data = [
        [
            "Filename",
            "Original_Title",
            "Language",
            "Epoch",
            "Area",
            "Lines",
            "Tokens",
        ]
    ]

    total_lines = 0
    total_tokens = 0

    for idx, doc in enumerate(tqdm(raw_docs, desc="Обработка")):
        title = doc.get("Title", "")
        if not title:
            title = doc.get("Original_title", f"doc_{idx}")

        content = doc.get("Content", "").strip()
        language = doc.get("Language", "Unknown").strip()
        epoch = doc.get("Epoch", "Unknown").strip()
        area = doc.get("Area", "Unknown").strip()

        if not content:
            continue

        safe_title = sanitize_filename(title)
        filename = f"{idx+1:03d}_{safe_title}.txt"
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        clean_line = clean_diacu_text(content)

        file_lines_count = 0
        file_tokens_count = 0

        if clean_line and len(clean_line) > 15 and len(clean_line.split()) > 1 and re.search(r"[а-яА-ЯёЁѣѢіІѵѴѫѸѡѠѕЅ]", clean_line):
            with open(output_path, "w", encoding="utf-8") as outfile:
                outfile.write(clean_line + "\n")

            file_lines_count = 1
            file_tokens_count = len(re.findall(r"\w+|[^\w\s]", clean_line))
        else:
            continue

        # Сохраняем стату, если в файле есть полезный текст
        if file_lines_count > 0:
            csv_data.append(
                [
                    filename,
                    title,
                    language,
                    epoch,
                    area,
                    file_lines_count,
                    file_tokens_count,
                ]
            )
            total_lines += file_lines_count
            total_tokens += file_tokens_count
        else:
            # Удаляем файл, если он оказался пустым после жесткой зачистки
            os.remove(output_path)

    # Сохраняем аналитику в CSV
    with open(CSV_OUTPUT_FILE, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)

    print("\n" + "=" * 60)
    print(f"🔥 ГОТОВО! Чистые тексты лежат в папке: {OUTPUT_FOLDER}")
    print(f"📊 Таблица для Google Sheets сохранена как: {CSV_OUTPUT_FILE}")
    print(f"Всего строк: {total_lines}")
    print(f"Всего токенов: {total_tokens}")
    print("=" * 60)


if __name__ == "__main__":
    main()
