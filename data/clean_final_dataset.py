import re
import os
from tqdm import tqdm  # Для красоты прогресс-бара

# --- НАСТРОЙКИ ---
INPUT_FILE = "final_dataset_ready.txt"
OUTPUT_FILE = "final_dataset_clean.txt"


def advanced_clean_text(text):
    if not isinstance(text, str):
        return ""

    # 1. Мусор
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"[\ue000-\uf8ff]", "", text)

    # 2. Гомоглифы (ОНИ-ТО И СЛОМАЛИ ТЕГИ!)
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

    # 3. Удаляем ударения, но ОСТАВЛЯЕМ титла
    text = re.sub(r"[\u0300-\u036f]", "", text)

    # 4. ЧИНИМ СЛОВА (Только если титла нет!)
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

    # 5. Пробелы
    text = re.sub(r"\s+", " ", text).strip()

    # --- ПАТЧ: ВОССТАНОВЛЕНИЕ ТЕГОВ ---
    # Мы ищем "испорченные" кириллицей теги и возвращаем их в латиницу.
    # Так как мы точно знаем, какие теги используем, проще всего сделать replace.

    # [СТХ_СНURСН] -> [CTX_CHURCH]
    text = text.replace("[СТХ_СНURСН]", "[CTX_CHURCH]")

    # [СТХ_LЕGАL] -> [CTX_LEGAL] (L, G остались латиницей, E, A стали кириллицей)
    text = text.replace("[СТХ_LЕGАL]", "[CTX_LEGAL]")

    # [СТХ_DАILY] -> [CTX_DAILY] (D, I, L, Y латиница, A кириллица)
    text = text.replace("[СТХ_DАILY]", "[CTX_DAILY]")

    # [СТХ_ВООК] -> [CTX_BOOK] (B, O, O, K стали кириллицей)
    text = text.replace("[СТХ_ВООК]", "[CTX_BOOK]")

    # [UNК] -> [UNK] (K стала кириллицей)
    text = text.replace("[UNК]", "[UNK]")

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
