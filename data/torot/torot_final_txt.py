import argparse
import glob
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# ОПРЕДЕЛЕНИЕ ЖАНРА
# ══════════════════════════════════════════════════════════════════════════════

def get_genre(filename: str) -> str:
    name = Path(filename).stem.lower()

    # CHURCH: церковные тексты, жития, сборники
    if any(word in name for word in [
        "kiev-mis", "psal-sin", "sergrad", "supr", "usp-sbor",
        "vit-const", "vit-meth", "zogr", "avv",
    ]):
        return "CHURCH"

    # DAILY: берестяные грамоты, письма, маргиналии, домострой
    elif any(word in name for word in [
        "birchbark", "domo", "kur", "mstislav-col", "mst",
        "nov-marg", "ostromir-col", "peter", "vest-kur",
    ]):
        return "DAILY"

    # LIT: летописи, повести, сказания, трактаты
    elif any(word in name for word in [
        "afnik", "const", "drac", "kiev-hyp", "lav", "luk-koloc",
        "nov-sin", "pskov", "pvl-hyp", "schism", "spi", "suz-lav", "zadon",
    ]):
        return "LIT"

    # Всё остальное — LEGAL
    else:
        return "LEGAL"


# ══════════════════════════════════════════════════════════════════════════════
# ОЧИСТКА ТЕКСТА (адаптировано из NKRYA-скрипта)
# ══════════════════════════════════════════════════════════════════════════════

TITLO_RANGE = range(0x0483, 0x0488)

PUNCT_MAP = {
    "†": "+", "×": "+", "*": "+",
    "⁘": ":", "⁙": ":", "⁞": ":", "¦": ":",
    "∙": "·", ".": "·", "҂": "·", "\uf13f": "·",
}

CYR_NUMERALS = "авгдєѕзиѳіклмнѯопрстуфхѱѡцчшщъыьѣюѧѩѫѷѵ"
NUM_PATTERN = re.compile(rf"([:+·])([{CYR_NUMERALS}]+҃)\1")


def _protect_numerals(text: str) -> tuple[str, dict]:
    protected: dict[str, str] = {}

    def repl(m: re.Match) -> str:
        key = f"PNUM{len(protected)}PNUM"
        protected[key] = m.group(0)
        return key

    return NUM_PATTERN.sub(repl, text), protected


def _unprotect_numerals(text: str, protected: dict) -> str:
    for key, val in protected.items():
        text = text.replace(key, val)
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Удаляем угловые скобки <...> вместе с содержимым
    text = re.sub(r"(<|‹)[^>]*(›|>)", " ", text)

    # Убираем скобки (), [], {}, оставляя содержимое
    text = re.sub(r"[(){}\[\]]", "", text)

    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    chars = []
    for ch in text:
        if ch in PUNCT_MAP:
            chars.append(PUNCT_MAP[ch])
            continue
        if ch in {"+", ":", "·"}:
            chars.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat[0] in {"L", "N", "M"} or ch.isspace():
            chars.append(ch)
        else:
            chars.append(" ")

    text = "".join(chars)

    # Приводим варианты титла к одному символу ҃
    text = "".join("҃" if ord(ch) in TITLO_RANGE else ch for ch in text)

    # Отделяем пунктуацию пробелами, защищая буквенные числительные
    text, protected_nums = _protect_numerals(text)
    text = re.sub(r"\s*([:+·])\s*", r" \1 ", text)
    text = _unprotect_numerals(text, protected_nums)

    # Убираем лишние combining marks, оставляем титло
    nfd = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn" or ch == "҃")
    text = unicodedata.normalize("NFC", text)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[.\s\-–—:]+", "", text)

    return text


def has_enough_cyrillic(text: str) -> bool:
    return len(re.findall(r"[а-яёѣіѵѫѡѕ]", text)) >= 3


def is_valid(text: str) -> bool:
    return (
            bool(text)
            and len(text) > 15
            and len(text.split()) > 1
            and has_enough_cyrillic(text)
    )


# ══════════════════════════════════════════════════════════════════════════════
# ПАРСЕРЫ
# ══════════════════════════════════════════════════════════════════════════════

def parse_conllu(filepath: str) -> list[str]:
    """
    Парсит .conllu (CoNLL-U) файл.
    Берёт форму токена из колонки 2 (индекс 1).
    Пропускает строки с многословными токенами (1-2, 1.1 и т.д.).
    """
    sentences: list[str] = []
    current: list[str] = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line or line.startswith("#"):
                if current:
                    cleaned = clean_text(" ".join(current))
                    if is_valid(cleaned):
                        sentences.append(cleaned)
                    current = []
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            token_id = parts[0]
            # Пропускаем многословные токены (1-2) и пустые узлы (1.1)
            if "-" in token_id or "." in token_id:
                continue

            form = parts[1]
            if form and form != "_":
                current.append(form)

    # Последнее предложение
    if current:
        cleaned = clean_text(" ".join(current))
        if is_valid(cleaned):
            sentences.append(cleaned)

    return sentences


def parse_xml(filepath: str) -> list[str]:
    """
    Парсит PROIEL XML (.xml) файл.
    Извлекает атрибут form из элементов <token>.
    Группирует по <sentence>.
    """
    sentences: list[str] = []

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  ⚠️  XML ошибка в {filepath}: {e}")
        return sentences

    # Namespace может быть пустым или задан
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    for sentence in root.iter(f"{ns}sentence"):
        tokens = []
        for token in sentence.iter(f"{ns}token"):
            form = token.get("form", "").strip()
            if form and form != "_":
                tokens.append(form)
        if tokens:
            cleaned = clean_text(" ".join(tokens))
            if is_valid(cleaned):
                sentences.append(cleaned)

    return sentences


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

CATEGORIES = ["DAILY", "CHURCH", "LIT", "LEGAL"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Парсер и фильтратор TOROT-датасета")
    parser.add_argument("--data_dir", default="./torot_data",
                        help="Папка с .conllu и/или .xml файлами TOROT")
    parser.add_argument("--out_dir", default="torot",
                        help="Папка для выходных файлов по категориям")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Открываем файлы-приёмники для каждой категории
    out_files = {
        cat: (out_dir / f"torot_{cat}.txt").open("w", encoding="utf-8")
        for cat in CATEGORIES
    }

    # Собираем все файлы
    conllu_files = sorted(data_dir.rglob("*.conllu")) + sorted(data_dir.rglob("*.conll"))
    xml_files = sorted(data_dir.rglob("*.xml"))
    all_files = conllu_files + xml_files

    print(f"📂 Найдено файлов: {len(conllu_files)} .conllu  +  {len(xml_files)} .xml")

    stats: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    skipped = 0

    for filepath in tqdm(all_files, desc="Парсинг"):
        suffix = filepath.suffix.lower()
        try:
            if suffix in {".conllu", ".conll"}:
                sentences = parse_conllu(str(filepath))
            elif suffix == ".xml":
                sentences = parse_xml(str(filepath))
            else:
                continue
        except Exception as e:
            print(f"  ❌ Ошибка {filepath.name}: {e}")
            skipped += 1
            continue

        if not sentences:
            skipped += 1
            continue

        genre = get_genre(str(filepath))
        document = " ".join(sentences)
        out_files[genre].write(document + "\n")
        stats[genre] += 1

    for f in out_files.values():
        f.close()

    # Удаляем пустые файлы
    for cat in CATEGORIES:
        p = out_dir / f"torot_{cat}.txt"
        if p.stat().st_size == 0:
            p.unlink()

    total = sum(stats.values())
    print("\n" + "=" * 55)
    print("✅ Готово!")
    print(f"   Пропущено файлов: {skipped}")
    print(f"   Всего документов: {total:,}")
    print()
    for cat, n in stats.items():
        if n:
            print(f"   torot_{cat}.txt  —  {n:,} документов")
    print("=" * 55)


if __name__ == "__main__":
    main()