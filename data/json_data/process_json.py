import json
import os
import re
import unicodedata

JSON_FILE = "DIACU_1.0.json"
OUTPUT_DIR = "diacu_extracted"

# Нормализуем только эти знаки пунктуации
# Всё остальное из пунктуации/символов будет удаляться или заменяться пробелом
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


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-_]", "_", name)
    return re.sub(r"_{2,}", "_", name)[:50]


def normalize_text(text: str) -> str:
    """
    Приводит текст к виду:
    - одна строка
    - нижний регистр
    - только + : · из пунктуации
    - остальные символы/пунктуация -> пробел
    - множественные пробелы схлопываются
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", str(text)).lower()
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    text = re.sub(r"\.{2,}", "[GAP]", text)
    text = text.replace("…", "[GAP]")

    out = []
    for ch in text:
        if ch in PUNCT_MAP:
            out.append(PUNCT_MAP[ch])
            continue

        if ch in {"+", ":", "·"}:
            out.append(ch)
            continue

        cat = unicodedata.category(ch)

        # Оставляем буквы, цифры, комбинирующие знаки (в т.ч. титло ҃),
        # а также пробелы
        if cat[0] in {"L", "N", "M"} or ch.isspace():
            out.append(ch)
        else:
            # Всё остальное (скобки, запятые, слэши, спецсимволы и т.п.)
            # заменяем на пробел
            out.append(" ")

    text = "".join(out)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    if not os.path.exists(JSON_FILE):
        print(f"File {JSON_FILE} does not exist.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Folder was created: {OUTPUT_DIR}")

    print("Reading JSON...")

    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = data.get("Documents", [])
        print(f"Found documents: {len(docs)}")

        for idx, doc in enumerate(docs):
            title = doc.get("Title", "Untitled")
            if not title:
                title = doc.get("Original_title", f"doc_{idx}")

            content = doc.get("Content", "")
            clean_content = normalize_text(content)

            safe_title = sanitize_filename(title)
            filename = f"{idx + 1:03d}_{safe_title}.txt"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, "w", encoding="utf-8") as out:
                out.write(clean_content + "\n")

        print(f"\nAll files are in the folder: {OUTPUT_DIR}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()