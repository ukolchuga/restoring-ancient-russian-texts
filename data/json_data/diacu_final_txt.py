import argparse
import json
import re
import unicodedata
from pathlib import Path
from bertislav_normalize import normalize

# =============================================================================
# CATEGORIZATION LOGIC
# =============================================================================

CATEGORIES = ["CHURCH", "DAILY", "LIT", "LEGAL"]

def get_category(title: str) -> str:
    """Classifies the document into a specific domain based on title keywords."""
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
        # Default fallback category
        return "LEGAL"


# =============================================================================
# TEXT CLEANING & NORMALIZATION
# =============================================================================

# Maps historical/rare symbols to standard structural punctuation
PUNCT_MAP = {
    "†": "+", "×": "+", "*": "+",
    "⁘": ":", "⁙": ":", "⁞": ":", "¦": ":",
    "∙": "·", ".": "·", "҂": "·", "\uf13f": "·",
    # Specific character normalizations
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

# Fragments indicating the start of modern editorial notes or metadata
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

# Placeholder used to protect gaps from being stripped out during number/symbol removal
_GAP_PLACEHOLDER = "_GAP_PLACEHOLDER_"


def cut_after_first_marker(text: str, markers: list[str]) -> str:
    """Truncates the text at the first occurrence of any known editorial marker."""
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
    """Determines if a document should be excluded from the final dataset."""
    source = str(doc.get("Source", "") or "")
    title_norm = unicodedata.normalize("NFC", str(title)).casefold().strip()

    if source.startswith(TOROTTREEBANK_PREFIX):
        return True

    if title_norm in TITLE_BLACKLIST:
        return True

    return False


def normalize_text(text: str) -> str:
    """Cleans, normalizes, and removes artifacts from the raw historical text."""
    if not text:
        return ""

    text = str(text)

    # 1) Truncate modern editorial footers/notes
    text = cut_after_first_marker(text, TRUNCATE_AFTER_MARKERS)

    # 2) Remove specific historical diacritics that impede processing (e.g., palatalization mark)
    text = text.replace("҆", "")

    # 3) Remove inline editorial expansions or page reconstructions found in {} or <>
    text = re.sub(r"\{.*?\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"<.*?>", " ", text, flags=re.DOTALL)

    # Normalize Unicode characters and flatten whitespace
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ").lower()

    # 4) Standardize ellipses indicating missing text into a distinct [GAP] token
    text = re.sub(r"\.{2,}|…", "[GAP]", text)

    # 5) Protect the [GAP] token from subsequent aggressive regex replacements
    text = text.replace("[GAP]", _GAP_PLACEHOLDER)

    # 6) Strip structural manuscript metadata and artifact numbers:
    
    # 6.1) Remove alphanumeric tags with periods at the end of words (e.g., ѡ...3. -> ѡ...)
    text = re.sub(r"(?<=[\u0400-\u052F\uA640-\uA69F])[0-9A-Za-z]+\.", "", text)

    # 6.2) Remove manuscript folio/page numbers (e.g., .162vне -> . не)
    text = re.sub(r"\.(\d+[rv])(?=[\u0400-\u052F\uA640-\uA69F])", ". ", text)

    # 6.3) Remove inline column/folio markers (e.g., въ|57бꙁвѣща́еть -> въꙁвѣща́еть)
    text = re.sub(r"\|(\d+[абab])(?=[\u0400-\u052F\uA640-\uA69F])", "", text)

    # 6.4) Remove Roman numerals followed by a dot (e.g., ХХХVІ.)
    text = re.sub(r"(?<!\w)[IVXLCDMХхVvІі]{2,}\.", " ", text)

    # 6.5) Remove all remaining standard digits (assumed to be archival noise)
    text = re.sub(r"\d+", " ", text)

    # 7) Character-level filtering:
    # Keep only letters, mapped punctuation, and whitespace. Replace everything else.
    out = []
    for ch in text:
        if ch in PUNCT_MAP:
            out.append(PUNCT_MAP[ch])
            continue
        if ch in {"+", ":", "·"}:
            out.append(ch)
            continue

        cat = unicodedata.category(ch)
        # Keep Letters (L), Numbers/Digits (N), Marks/Diacritics (M), and Spaces
        if cat[0] in {"L", "N", "M"} or ch.isspace():
            out.append(ch)
        else:
            # Replaces unmapped punctuation (like standalone brackets) with spaces
            out.append(" ")

    text = "".join(out)

    # 8) Restore the protected [GAP] tokens
    text = text.replace(_GAP_PLACEHOLDER, "[GAP]")

    # 9) Collapse multiple spaces into a single space and strip trailing/leading whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Apply external NLP library normalization (e.g., handling specific orthography)
    text = normalize(text)

    return text


def has_enough_cyrillic(text: str) -> bool:
    """Ensures the document contains a minimum threshold of valid historical Cyrillic letters."""
    return len(re.findall(r"[а-яёѣіѵѫѡѕ]", text)) >= 3


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="DIACU Dataset Parser and Categorizer")
    parser.add_argument("--json", default="DIACU_1.0.json", help="Path to input JSON dataset")
    parser.add_argument("--out_dir", default=".", help="Directory for output TXT files")
    args = parser.parse_args()

    json_path = Path(args.json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"Error: File not found at {json_path}")
        return

    print(f"Reading dataset: {json_path}...")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    docs = data.get("Documents", [])
    print(f"Documents found in JSON: {len(docs):,}")

    # Prepare output file streams for each category
    out_files = {
        cat: (out_dir / f"diacu_{cat}.txt").open("w", encoding="utf-8")
        for cat in CATEGORIES
    }
    
    stats: dict = {cat: 0 for cat in CATEGORIES}
    skipped = 0

    # Process each document
    for idx, doc in enumerate(docs):
        title = doc.get("Title", "") or doc.get("Original_title", f"doc_{idx}")

        if should_skip_doc(doc, title):
            skipped += 1
            continue

        content = doc.get("Content", "")
        cleaned = normalize_text(content)

        # Skip empty documents or those that lack sufficient Cyrillic content after cleaning
        if not cleaned or not has_enough_cyrillic(cleaned):
            skipped += 1
            continue

        cat = get_category(title)
        out_files[cat].write(cleaned + "\n")
        stats[cat] += 1

    # Close all file streams
    for f in out_files.values():
        f.close()

    # Cleanup: Remove any output files that remained empty
    for cat in CATEGORIES:
        p = out_dir / f"diacu_{cat}.txt"
        if p.stat().st_size == 0:
            p.unlink()

    # Print processing summary
    total = sum(stats.values())
    labels = {
        "CHURCH": "Church (Religious)",
        "DAILY":  "Daily (Everyday)",
        "LIT":    "Literary",
        "LEGAL":  "Legal/Admin",
    }
    
    print("\n" + "=" * 55)
    print("Processing Complete!")
    print(f"  Documents Skipped: {skipped:,}")
    print(f"  Documents Written: {total:,}")
    print()
    for cat in CATEGORIES:
        if stats[cat]:
            print(f"  {labels[cat]:<18} ({cat}): {stats[cat]:>5,}  -> diacu_{cat}.txt")
    print("=" * 55)


if __name__ == "__main__":
    main()