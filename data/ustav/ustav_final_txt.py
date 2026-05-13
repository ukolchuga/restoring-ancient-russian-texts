from pathlib import Path
import re
import unicodedata

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ustav_for_training.txt"
OUTPUT_FILE = BASE_DIR / "ustav_final_cleaned.txt"

TITLO_RANGE = range(0x0483, 0x0488)

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

# Древнерусские буквенные числительные вида :л҃: или ·в·
CYR_NUMERALS = "авгдєѕзиѳіклмнѯопрстуфхѱѡцчшщъыьѣюѧѩѫѷѵ"
NUM_PATTERN = re.compile(rf"([:+·])([{CYR_NUMERALS}]+҃?)\1")
CTX_RE = re.compile(r"^\[CTX_[A-Z_]+\]\s*")
GAP_RE = re.compile(r"\[gap\]", flags=re.IGNORECASE)


def _protect_gaps(text: str):
    protected = {}

    def repl(m):
        key = f"ggg{len(protected)}ggg"
        protected[key] = "[GAP]"
        return key

    return GAP_RE.sub(repl, text), protected


def _unprotect_gaps(text: str, protected: dict[str, str]) -> str:
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


def _protect_numerals(text: str):
    protected = {}

    def repl(m):
        key = f"PNUM{len(protected)}PNUM"
        protected[key] = m.group(0)
        return key

    return NUM_PATTERN.sub(repl, text), protected


def _unprotect_numerals(text: str, protected: dict[str, str]) -> str:
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFC", str(text)).strip()

    # Удаляем ссылки
    text = re.sub(r"\[\s*\d+\s*\]", "", text)

    # Удаляем разметку в угловых скобках целиком
    text = re.sub(r"(<|‹)[^>]*(›|>)", " ", text)
    text = re.sub(r"\(.*?\)|\[.*?\]|(<|‹).*?(›|>)", " ", text)

    # Удаляем только фигурные скобки
    text = re.sub(r"[{}]", "", text)

    # Нижний регистр
    text = text.lower()

    # Приводим все варианты титла к одному символу ҃
    text = "".join("҃" if ord(ch) in TITLO_RANGE else ch for ch in text)

    # Нормализация пунктуации
    chars = []
    for ch in text:
        if ch in PUNCT_MAP:
            chars.append(PUNCT_MAP[ch])
            continue

        if ch in {"+", ":", "·"}:
            chars.append(ch)
            continue

        cat = unicodedata.category(ch)
        # Сохраняем буквы, цифры, пробелы, квадратные скобки и combining marks
        if cat[0] in {"L", "N", "M"} or ch.isspace() or ch in "()[]":
            chars.append(ch)
        else:
            chars.append(" ")

    text = "".join(chars)

    # Защищаем древнерусские числительные
    text, protected_nums = _protect_numerals(text)

    # Отделяем пунктуацию пробелами
    text = re.sub(r"\s*([:+·])\s*", r" \1 ", text)

    # Возвращаем числительные
    text = _unprotect_numerals(text, protected_nums)

    # Убираем лишние combining marks
    nfd = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn" or ch == "҃")
    text = unicodedata.normalize("NFC", text)

    # Схлопываем пробелы
    text = re.sub(r"\s+", " ", text).strip()

    # Чистим мусор в начале
    text = re.sub(r"^[.\s\-\–\—\:]+", "", text)

    # Заменяем [gap] на [GAP]
    text = re.sub(r"\[gap\]", "[GAP]", text, flags=re.IGNORECASE)

    # Защищаем [GAP], затем удаляем остальные скобки
    text = text.replace("[GAP]", "GAPPROTECTED")
    text = re.sub(r"[(){}\[\]]", "", text)
    text = text.replace("GAPPROTECTED", "[GAP]")

    return text
    return text


def main():
    if not INPUT_FILE.exists():
        print(f"❌ Файл не найден: {INPUT_FILE}")
        return

    lines = INPUT_FILE.read_text(encoding="utf-8").splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        clean_line = clean_text(line)
        if clean_line:
            cleaned.append(clean_line)

    OUTPUT_FILE.write_text(" ".join(cleaned) + "\n", encoding="utf-8")
    print(f"✅ Сохранено: {OUTPUT_FILE}")
    print(f"📄 Строк: {len(cleaned)}")


if __name__ == "__main__":
    main()