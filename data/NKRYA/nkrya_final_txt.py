from pathlib import Path
import re
import unicodedata

BASE_DIR = Path(__file__).resolve().parent
INPUT_ROOT = BASE_DIR / "NKRYA_TEXTS"

FOLDERS = [
    "DAILY",
    "LEGAL",
    "LIT",
    "SCIENCE",
]

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

CYR_NUMERALS = "авгдєѕзиѳіклмнѯопрстуфхѱѡцчшщъыьѣюѧѩѫѷѵ"
NUM_PATTERN = re.compile(rf"([:+·])([{CYR_NUMERALS}]+҃)\1")


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

    # Удаляем <...> вместе с содержимым
    text = re.sub(r"(<|‹)[^>]*(›|>)", " ", text)

    # Сначала нормализуем каверзные варианты gap'а в специальный placeholder,
    # чтобы последующее удаление скобок не уничтожило маркер.
    # Ловим: gap, GAP PLACEHOLDER, [gap], с любым регистром и с/без слова "placeholder".
    GAP_PLACEHOLDER = "__GAP_PLACEHOLDER__"
    text = re.sub(r"\[?\b(?:gap(?:\s*placeholder)?)\b\]?", GAP_PLACEHOLDER, text, flags=re.IGNORECASE)

    # Удаляем скобки (), [], {}, оставляя содержимое (placeholder не содержит скобок, поэтому сохранится)
    text = re.sub(r"[(){}\[\]]", "", text)

    text = unicodedata.normalize("NFC", str(text)).lower()
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

        # Сохраняем буквы, цифры, пробелы и combining marks,
        # чтобы титло и похожие знаки не терялись
        if cat[0] in {"L", "N", "M"} or ch.isspace():
            chars.append(ch)
        else:
            chars.append(" ")

    text = "".join(chars)

    # Приводим все варианты титла к одному символу ҃
    text = "".join("҃" if ord(ch) in TITLO_RANGE else ch for ch in text)

    # Отделяем пунктуацию пробелами, но не ломаем буквенные числительные
    text, protected_nums = _protect_numerals(text)
    text = re.sub(r"\s*([:+·])\s*", r" \1 ", text)
    text = _unprotect_numerals(text, protected_nums)

    # Убираем лишние combining marks, но оставляем титло
    nfd = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn" or ch == "҃")
    text = unicodedata.normalize("NFC", text)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[.\s\-\–\—\:]+", "", text)

    # Восстанавливаем placeholder в единый маркер [GAP] (в верхнем регистре),
    # делаем замену регистронезависимо (placeholder ранее мог быть пониженного регистра).
    text = re.sub(re.escape(GAP_PLACEHOLDER), "[GAP]", text, flags=re.IGNORECASE)

    return text


def has_enough_cyrillic(text: str) -> bool:
    return len(re.findall(r"[а-яА-ЯёЁѣѢіІѵѴѫѸѡѠѕЅ]", text)) >= 3


def main():
    for folder_name in FOLDERS:
        folder = INPUT_ROOT / folder_name
        output_file = INPUT_ROOT / f"NKRYA_{folder_name}.txt"

        if not folder.exists():
            print(f"⚠️ Папка не найдена: {folder}")
            continue

        txt_files = sorted(folder.rglob("*.txt"))
        print(f"📂 {folder_name}: {len(txt_files)} файлов")

        written = 0
        with output_file.open("w", encoding="utf-8") as out:
            for file_path in txt_files:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                text = clean_text(text)

                if not text:
                    continue

                # Аналогично all.py: отбрасываем мусорные/слишком короткие тексты
                if len(text) <= 15 or len(text.split()) <= 1 or not has_enough_cyrillic(text):
                    continue

                out.write(text + "\n")
                written += 1

        print(f"✅ Сохранено: {output_file} — {written} документов")


if __name__ == "__main__":
    main()