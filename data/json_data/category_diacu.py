import os
import shutil


INPUT_FOLDER = "diacu_cleaned_texts"

FOLDERS = {
    "CHURCH": "diacu_CHURCH",
    "DAILY": "diacu_DAILY",
    "LIT": "diacu_LIT",
    "LEGAL": "diacu_LEGAL",
}


def get_category(filename):
    name = filename.lower()

    if any(
        word in name
        for word in [
            "evangelie",
            "psaltir",
            "slovo",
            "pooučenie",
            "zlatostruj",
            "žitie",
            "molitva",
            "apostol",
            "služba",
            "kanon",
            "pochvala",
            "šestodnev",
            "vita",
            "dioptra",
            "sbornik",
            "prolog",
            "triod",
            "missal",
            "life",
            "житие",
            "zapovědan",
            "pamjat",
            "mltva",
            "mlitvy",
            "bogoslovie",
            "damaskin",
            "service",
            "trebnik",
            "sinodik",
            "шестоднев",
            "avva",
        ]
    ):
        return "CHURCH"
    elif any(
        word in name
        for word in [
            "birchbark",
            "domostroj",
            "otpiska",
            "poslanie",
            "zagovor",
            "gramotki",
            "correspondence",
            "missive",
            "colophon",
            "marginalia",
            "kuranty",
            "prayer",
            "vesti",
        ]
    ):
        return "DAILY"
    elif any(
        word in name
        for word in [
            "povest",
            "pověst",
            "tale",
            "skazanie",
            "chronicle",
            "chronika",
            "istorija",
            "zadonščina",
            "journey",
            "stepennaja",
            "stepennaja",
            "komidija",
            "pritči",
            "dialozi",
            "history",
        ]
    ):
        return "LIT"
    else:
        return "LEGAL"


def main():
    for folder in FOLDERS.values():
        os.makedirs(folder, exist_ok=True)

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]
    print(f"Sort start {len(files)} files...\n")
    categorized_files = {"CHURCH": [], "DAILY": [], "LIT": [], "LEGAL": []}

    for filename in files:
        category = get_category(filename)
        source_path = os.path.join(INPUT_FOLDER, filename)
        target_path = os.path.join(FOLDERS[category], filename)

        shutil.copy2(source_path, target_path)
        categorized_files[category].append(filename)

    print("\n" + "=" * 50)
    print("📋 РАСПРЕДЕЛЕНИЕ ФАЙЛОВ ПО ТЕГАМ")
    print("=" * 50)

    for cat, file_list in categorized_files.items():
        print(f"\n📂 Категория [{cat}] — {len(file_list)} файлов:")
        for f in file_list:
            print(f"  - {f}")

    print("\n" + "=" * 50)
    print("=== ИТОГОВАЯ СТАТИСТИКА ===")
    print(f"⛪️ Церковные (CHURCH): {len(categorized_files['CHURCH'])} файлов")
    print(f"🏡 Бытовые (DAILY): {len(categorized_files['DAILY'])} файлов")
    print(f"📚 Литературные (LIT): {len(categorized_files['LIT'])} файлов")
    print(f"📜 Юридические (LEGAL / Остаток): {len(categorized_files['LEGAL'])} файлов")
    print("=" * 50)


if __name__ == "__main__":
    main()
