import argparse
import json
import re
import unicodedata
from pathlib import Path
from bertislav_normalize import normalize

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
    # буквы
    "": "ч", "": "с҃", "": "и", "вⷬ": "вр",
}

TITLE_BLACKLIST = {
    "druga molitva za rodilka",
    "molitva za rodilka sled raždaneto",
    "žitie vassiana tiksnenskogo (žitie i čudesa 1–8)",
    "žitie feodosija totemskogo",
    "žitie stefanа komel'skogo",
    "žitie kassiana ugličskogo",
    "žitie innokentija komel'skogo",
    "žitie ignatija vologodskogo",
    "žitie ignatija lomskogo",
    "žitie gerasima vologodskogo",
    "žitie arsenija komel'skogo",
    "žitie andreja totemskogo",
    "žitie aleksandra kuštskogo i evfimija sjanžemskogo",
    "žitie aleksandra kuštskogo",
    "житие вассиана тиксненского (чудеса 9–29)"
}

TRUNCATE_AFTER_MARKERS = [
    "С наало  богословленеС",
    "Заглавието е по НБКМ",
    "В ркп.",
    "текѫщи Т.кѵрь Т.",
    "Загубено е началото на текста с обем един лист",
    "Вѣдѣнїе Z : Вѣданїе Sбы́въшхъ Z",
    "Johannis Chrysostomi homilia in annuntiationem Deiparae",
    "Заглавието е по Михайловския препис от ЦНБ",
    "Разликата между оргиналния",
    "Словото за света Троица е познато в две негови редакции",
    "ПИ́САСЕ ПО Е̑ЛꙿЛИКО ПИ́СМЕНА",
]

TOROTTREEBANK_PREFIX = "https://github.com/torottreebank/"

# Важно: без цифр — чтобы потом не сломался при удалении чисел
_GAP_PLACEHOLDER = "_GAP_PLACEHOLDER_"


def cut_after_first_marker(text: str, markers: list[str]) -> str:
    """Обрезает текст по первому найденному маркеру."""
    first_pos = None
    for marker in markers:
        m = re.search(re.escape(marker), text, flags=re.IGNORECASE)
        if m:
            pos = m.start()
            if first_pos is None or pos < first_pos:
                first_pos = pos
    if first_pos is None:
        return text
    return text[:first_pos]


def should_skip_doc(doc: dict, title: str) -> bool:
    source = str(doc.get("Source", "") or "")
    title_norm = unicodedata.normalize("NFC", str(title)).casefold().strip()

    if source.startswith(TOROTTREEBANK_PREFIX):
        return True

    if title_norm in TITLE_BLACKLIST:
        return True

    return False



def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = str(text)

    # 1) Обрезаем текст после служебных фрагментов
    text = cut_after_first_marker(text, TRUNCATE_AFTER_MARKERS)

    # 2) Убираем "҆"
    text = text.replace("҆", "")

    # 3) Удаляем содержимое {} и <>
    text = re.sub(r"\{.*?\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"<.*?>", " ", text, flags=re.DOTALL)

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ").lower()

    # 4) Многоточия -> [GAP]
    text = re.sub(r"\.{2,}|…", "[GAP]", text)

    # 5) Защищаем [GAP] перед общими заменами
    text = text.replace("[GAP]", _GAP_PLACEHOLDER)

    # 6) Убираем числительные/мусор по вашим правилам

    # 6.1) цифры/латиница в конце слова с точкой: ѡ...3. -> ѡ...
    text = re.sub(
        r"(?<=[\u0400-\u052F\uA640-\uA69F])[0-9A-Za-z]+\.",
        "",
        text,
    )

    # 6.2) .162vне -> . не
    text = re.sub(
        r"\.(\d+[rv])(?=[\u0400-\u052F\uA640-\uA69F])",
        ". ",
        text,
    )

    # 6.3) въ|57бꙁвѣща́еть -> въꙁвѣща́еть
    #      прѣи... . |57аѡ͗ -> прѣи... . ѡ͗
    text = re.sub(
        r"\|(\d+[абab])(?=[\u0400-\u052F\uA640-\uA69F])",
        "",
        text,
    )

    # 6.4) Римские числительные с точкой в конце: ХХХVІ. -> ""
    text = re.sub(
        r"(?<!\w)[IVXLCDMХхVvІі]{2,}\.",
        " ",
        text,
    )

    # 6.5) Все остальные числа -> пробелы
    text = re.sub(r"\d+", " ", text)

    # 7) Общая посимвольная фильтрация:
    #    - сохраняем буквы, цифры (если остались), знаки-маркеры, пробелы
    #    - всё остальное -> пробел
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
            # скобки тоже сюда попадут: они удалятся,
            # а содержимое останется
            out.append(" ")

    text = "".join(out)


    # 8) Возвращаем [GAP]
    text = text.replace(_GAP_PLACEHOLDER, "[GAP]")

    # 9) Убираем лишние пробелы
    text = re.sub(r"\s+", " ", text).strip()

    text = normalize(text)

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

        if should_skip_doc(doc, title):
            skipped += 1
            continue

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