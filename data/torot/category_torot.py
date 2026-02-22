import os
import shutil

INPUT_FOLDER = "torot_cleaned_texts"  # Укажи здесь папку с файлами TOROT

FOLDERS = {
    "CHURCH": "torot_CHURCH",
    "DAILY": "torot_DAILY",
    "LIT": "torot_LIT",
    "LEGAL": "torot_LEGAL",
}


def get_torot_category(filename):
    name = filename.lower()

    # 1. CHURCH: Церковные тексты, жития, сборники
    if any(
        word in name
        for word in [
            "kiev-mis",
            "psal-sin",
            "sergrad",
            "supr",
            "usp-sbor",
            "vit-const",
            "vit-meth",
            "zogr",
            "avv",
        ]
    ):
        return "CHURCH"

    # 2. DAILY: Берестяные грамоты, письма, маргиналии, домострой
    elif any(
        word in name
        for word in [
            "birchbark",
            "domo",
            "kur",
            "mstislav-col",
            "mst",
            "nov-marg",
            "ostromir-col",
            "peter",
            "vest-kur",
        ]
    ):
        return "DAILY"

    # 3. LIT: Летописи, повести, сказания, трактаты
    elif any(
        word in name
        for word in [
            "afnik",
            "const",
            "drac",
            "kiev-hyp",
            "lav",
            "luk-koloc",
            "nov-sin",
            "pskov",
            "pvl-hyp",
            "schism",
            "spi",
            "suz-lav",
            "zadon",
        ]
    ):
        return "LIT"

    else:
        return "LEGAL"


def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Ошибка: Папка {INPUT_FOLDER} не найдена.")
        return

    for folder in FOLDERS.values():
        os.makedirs(folder, exist_ok=True)

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]
    print(f"🚀 Начинаем сортировку {len(files)} файлов TOROT...\n")

    categorized_files = {"CHURCH": [], "DAILY": [], "LIT": [], "LEGAL": []}

    for filename in files:
        category = get_torot_category(filename)
        source_path = os.path.join(INPUT_FOLDER, filename)
        target_path = os.path.join(FOLDERS[category], filename)

        shutil.copy2(source_path, target_path)
        categorized_files[category].append(filename)

    # === НАГЛЯДНЫЙ ВЫВОД СПИСКОВ ФАЙЛОВ ===
    print("\n" + "=" * 50)
    print("📋 РАСПРЕДЕЛЕНИЕ ФАЙЛОВ TOROT ПО ТЕГАМ")
    print("=" * 50)

    for cat, file_list in categorized_files.items():
        print(f"\n📂 Категория [{cat}] — {len(file_list)} файлов:")
        for f in file_list:
            print(f"  - {f}")

    # === ИТОГОВАЯ СТАТИСТИКА ===
    print("\n" + "=" * 50)
    print("=== ИТОГОВАЯ СТАТИСТИКА TOROT ===")
    print(f"⛪️ Церковные (CHURCH): {len(categorized_files['CHURCH'])} файлов")
    print(f"🏡 Бытовые (DAILY): {len(categorized_files['DAILY'])} файлов")
    print(f"📚 Литературные (LIT): {len(categorized_files['LIT'])} файлов")
    print(f"📜 Юридические (LEGAL): {len(categorized_files['LEGAL'])} файлов")
    print("=" * 50)


if __name__ == "__main__":
    main()
