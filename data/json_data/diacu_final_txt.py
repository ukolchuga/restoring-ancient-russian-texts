import argparse
import json
import re
import unicodedata
from pathlib import Path

# =============================================================================
# ОПРЕДЕЛЕНИЕ КАТЕГОРИИ
# =============================================================================

CATEGORIES = ["CHURCH", "DAILY", "LIT", "LEGAL"]


def get_category(title: str) -> str:
    name = title.lower()

    if any(word in name for word in [
        "evangelie", "psaltir", "slovo", "pooučenie", "zlatostruj", "žitie",
        "molitva", "apostol", "služba", "kanon", "pochvala", "šestodnev",
        "vita", "dioptra", "sbornik", "prolog", "triod", "missal", "life",
        "житие", "zapovědan", "pamjat", "mltva", "mlitvy", "bogoslovie",
        "damaskin", "service", "trebnik", "sinodik", "avva",
    ]):
        return "CHURCH"

    elif any(word in name for word in [
        "birchbark", "domostroj", "otpiska", "poslanie", "zagovor",
        "gramotki", "correspondence", "missive", "colophon", "marginalia",
        "kuranty", "prayer", "vesti",
    ]):
        return "DAILY"

    elif any(word in name for word in [
        "povest", "pověst", "tale", "skazanie", "chronicle", "chronika",
        "istorija", "zadonščina", "journey", "stepennaja", "komidija",
        "pritči", "dialozi", "history",
    ]):
        return "LIT"

    else:
        return "LEGAL"


# =============================================================================
# ОЧИСТКА ТЕКСТА
# =============================================================================

PUNCT_MAP = {
    "†": "+", "×": "+", "*": "+",
    "⁘": ":", "⁙": ":", "⁞": ":", "¦": ":",
    "∙": "·", ".": "·", "҂": "·", "\uf13f": "·",
}

_GAP_PLACEHOLDER = "GAPTOKEN99"


def normalize_text(text: str) -> str:
    if not text:
        return ""

    # Удаляем СлС
    # 1. СлС — всегда маркер, удаляем везде
    text = re.sub(r"СлС", "", text)

    # 2. Сл — маркер, только если не продолжается строчными кириллическими
    #    (чтобы не задеть начало слова типа "Слово")
    text = re.sub(r"Сл(?![а-яёѣіѵѫѡѕ\u0430-\u052F\uA640-\uA69F])", "", text)

    # 3. Одиночная С — маркер, только если не окружена строчными кириллическими
    #    (чтобы не задеть С внутри слов типа "съ", "сего")
    text = re.sub(r"(?<![а-яёѣіѵѫѡѕ\u0430-\u052F\uA640-\uA69F])С(?![а-яёѣіѵѫѡѕ\u0430-\u052F\uA640-\uA69F])", "", text)

    text = unicodedata.normalize("NFC", str(text)).lower()
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # Многоточия -> [GAP] до посимвольной обработки
    text = re.sub(r"\.{2,}", "[GAP]", text)
    text = text.replace("\u2026", "[GAP]")

    # Защищаем [GAP] — иначе квадратные скобки удалятся в цикле
    text = text.replace("[GAP]", _GAP_PLACEHOLDER)

    # Удаляем ссылки
    text = re.sub(r"(?<=[\u0400-\u052F\uA640-\uA69F])\d+[a-z]*", "", text)

    out = []
    for ch in text:
        if ch in PUNCT_MAP:
            out.append(PUNCT_MAP[ch])
            continue
        if ch in {"+", ":", "·"}:
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat[0] in {"L", "N", "M"} or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")

    text = "".join(out)
    text = text.replace(_GAP_PLACEHOLDER, "[GAP]")
    # text = re.sub(r"(\s*\[GAP\]\s*)+", "[GAP]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_enough_cyrillic(text: str) -> bool:
    return len(re.findall(r"[а-яёѣіѵѫѡѕ]", text)) >= 3


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Парсер и категоризатор DIACU")
    parser.add_argument("--json", default="DIACU_1.0.json")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    json_path = Path(args.json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"Файл не найден: {json_path}")
        return

    print(f"Читаем {json_path}...")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    docs = data.get("Documents", [])
    print(f"  Документов в JSON: {len(docs):,}")

    out_files = {
        cat: (out_dir / f"diacu_{cat}.txt").open("w", encoding="utf-8")
        for cat in CATEGORIES
    }
    stats: dict = {cat: 0 for cat in CATEGORIES}
    skipped = 0

    for idx, doc in enumerate(docs):
        title = doc.get("Title", "") or doc.get("Original_title", f"doc_{idx}")
        content = doc.get("Content", "")

        cleaned = normalize_text(content)
        if not cleaned or not has_enough_cyrillic(cleaned):
            skipped += 1
            continue

        cat = get_category(title)
        out_files[cat].write(cleaned + "\n")
        stats[cat] += 1

    for f in out_files.values():
        f.close()

    # Удаляем пустые файлы
    for cat in CATEGORIES:
        p = out_dir / f"diacu_{cat}.txt"
        if p.stat().st_size == 0:
            p.unlink()

    total = sum(stats.values())
    labels = {
        "CHURCH": "Церковные",
        "DAILY":  "Бытовые",
        "LIT":    "Литературные",
        "LEGAL":  "Юридические",
    }
    print("\n" + "=" * 55)
    print("Готово!")
    print(f"  Пропущено: {skipped}")
    print(f"  Записано:  {total:,}")
    print()
    for cat in CATEGORIES:
        if stats[cat]:
            print(f"  {labels[cat]:<15} ({cat}): {stats[cat]:>5,}  -> diacu_{cat}.txt")
    print("=" * 55)


if __name__ == "__main__":
    main()