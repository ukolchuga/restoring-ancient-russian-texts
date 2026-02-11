import re
import os
from tqdm import tqdm  # Для красоты прогресс-бара

# --- НАСТРОЙКИ ---
INPUT_FILE = "final_dataset_ready.txt"
OUTPUT_FILE = "final_dataset_clean.txt"


def advanced_clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.replace("\ufeff", "").replace("\u200b", "")

    text = re.sub(r"[\ue000-\uf8ff]", "", text)

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


    text = re.sub(r"[\u0300-\u036f]", "", text)


    abbrev_map = {
        r"\bбг\b": "богъ",
        r"\bгд\b": "господь",
        r"\bсн\b": "сынъ",
        r"\bхс\b": "христосъ",
    }
    for pattern, repl in abbrev_map.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)


    text = re.sub(r"\s+", " ", text).strip()

    return text


print(f">>> 🧼 Начинаем генеральную уборку файла {INPUT_FILE}...")

if not os.path.exists(INPUT_FILE):
    print(
        f"❌ Ошибка: Файл {INPUT_FILE} не найден! Сначала создайте его предыдущим скриптом."
    )
else:
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, open(
        OUTPUT_FILE, "w", encoding="utf-8"
    ) as f_out:

        lines = f_in.readlines()
        cleaned_count = 0

        for line in tqdm(lines, desc="Обработка строк"):
            # Пропускаем разделители разделов (чтобы сохранить структуру)
            if line.startswith("---"):
                f_out.write(line)
                continue

            original = line
            cleaned = advanced_clean_text(line)

            # Если строка не пустая после чистки - записываем
            if len(cleaned) > 1:
                f_out.write(cleaned + "\n")
                cleaned_count += 1

    print(f"\n✅ ГОТОВО! Чистый файл: {OUTPUT_FILE}")
    print(f"📊 Сохранено строк: {cleaned_count}")
    print("Теперь можно обучать токенизатор и модель заново!")
