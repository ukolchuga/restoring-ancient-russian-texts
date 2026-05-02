#!/usr/bin/env python3
"""
prepare_splits.py

Читает исходные файлы по SOURCES_CONFIG, применяет теги и веса,
очищает текст, затем делит данные на три части:

  splits/train.txt     — основной корпус (95%), стратифицировано по CTX-тегу
  splits/test_a.txt    — основной корпус (5%), искусственные пропуски генерирует коллатор
  splits/test_b.jsonl  — источники со скобками; (text) и [text] -> [MASK]xlen

Поле "role" в SOURCES_CONFIG:
  "train"  — строки идут в train/test_a (веса применяются)
  "test_b" — строки со скобками идут в test_b; без скобок — опционально в train
             (при --charters_to_train строки без скобок добавляются в train с weight)

Формат test_b.jsonl:
  {
    "original":       "[CTX_DAILY] ѿ (кꙑꙗса) ї ...",
    "masked_input":   "[CTX_DAILY] ѿ [MASK][MASK][MASK][MASK][MASK][MASK] ї ...",
    "target":         "[CTX_DAILY] ѿ кꙑꙗса ї ...",
    "tag":            "[CTX_DAILY]",
    "n_masked_chars": 6,
    "spans":          [{"text": "кꙑꙗса", "type": "round"}]
  }

Запуск:
  python data/prepare_splits.py --out_dir splits
  python data/prepare_splits.py --out_dir splits --no_square --charters_to_train
"""

import argparse
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# КОНФИГУРАЦИЯ ИСТОЧНИКОВ
# ============================================================================

SOURCES_CONFIG = [
    # -- Daily ----------------------------------------------------------------
    {
        "type": "file",
        "path": "data/birch_bark/gramoty_final_categories_with_brackets.txt",
        "tag": "",
        "weight": 40,
        "role": "test_b",
    },
    {
        "type": "folder",
        "path": "data/json_data/diacu_DAILY",
        "tag": "[CTX_DAILY]",
        "weight": 15,
        "role": "train",
    },
    {
        "type": "folder",
        "path": "data/torot/torot_DAILY",
        "tag": "[CTX_DAILY]",
        "weight": 15,
        "role": "train",
    },
    {
        "type": "folder",
        "path": "data/pushkinskij_texts/clean_texts/DAILY",
        "tag": "[CTX_DAILY]",
        "weight": 15,
        "role": "train",
    },
    {
        "type": "folder",
        "path": "data/NKRYA/NKRYA_TEXTS/DAILY",
        "tag": "[CTX_DAILY]",
        "weight": 15,
        "role": "train",
    },
    # -- Church ---------------------------------------------------------------
    {
        "type": "file",
        "path": "data/epigraphica/epigraphica_final_cleaned_with_brackets.txt",
        "tag": "",
        "weight": 20,
        "role": "test_b",
    },
    {
        "type": "file",
        "path": "data/bible/bible_full_clean.txt",
        "tag": "[CTX_CHURCH]",
        "weight": 1,
        "role": "train",
    },
    {
        "type": "folder",
        "path": "data/json_data/diacu_CHURCH",
        "tag": "[CTX_CHURCH]",
        "weight": 1,
        "role": "train",
    },
    {
        "type": "folder",
        "path": "data/torot/torot_CHURCH",
        "tag": "[CTX_CHURCH]",
        "weight": 1,
        "role": "train",
    },
    {
        "type": "folder",
        "path": "data/pushkinskij_texts/clean_texts/CHURCH",
        "tag": "[CTX_CHURCH]",
        "weight": 1,
        "role": "train",
    },
    # -- Literature -----------------------------------------------------------
    {
        "type": "folder",
        "path": "data/json_data/diacu_LIT",
        "tag": "[CTX_LIT]",
        "weight": 3,
        "role": "train",
    },
    {
        "type": "folder",
        "path": "data/torot/torot_LIT",
        "tag": "[CTX_LIT]",
        "weight": 5,
        "role": "train",
    },
    {
        "type": "folder",
        "path": "data/pushkinskij_texts/clean_texts/LIT",
        "tag": "[CTX_LIT]",
        "weight": 4,
        "role": "train",
    },
    {
        "type": "folder",
        "path": "data/NKRYA/NKRYA_TEXTS/LIT",
        "tag": "[CTX_LIT]",
        "weight": 4,
        "role": "train",
    },
    # -- Legal ----------------------------------------------------------------
    {
        "type": "folder",
        "path": "data/json_data/diacu_LEGAL",
        "tag": "[CTX_LEGAL]",
        "weight": 2,
        "role": "train",
    },
    {
        "type": "folder",
        "path": "data/torot/torot_LEGAL",
        "tag": "[CTX_LEGAL]",
        "weight": 20,
        "role": "train",
    },
    {
        "type": "folder",
        "path": "data/pushkinskij_texts/clean_texts/LEGAL",
        "tag": "[CTX_LEGAL]",
        "weight": 10,
        "role": "train",
    },
    {
        "type": "folder",
        "path": "data/NKRYA/NKRYA_TEXTS/LEGAL",
        "tag": "[CTX_LEGAL]",
        "weight": 10,
        "role": "train",
    },
    # -- Epic -----------------------------------------------------------------
    {
        "type": "folder",
        "path": "data/bilini",
        "tag": "[CTX_EPIC]",
        "weight": 10,
        "role": "train",
    },
    {
        "type": "folder",
        "path": "data/pushkinskij_texts/clean_texts/EPIC",
        "tag": "[CTX_EPIC]",
        "weight": 10,
        "role": "train",
    },
    # -- Science --------------------------------------------------------------
    {
        "type": "folder",
        "path": "data/pushkinskij_texts/clean_texts/SCIENCE",
        "tag": "[CTX_SCIENCE]",
        "weight": 50,
        "role": "train",
    },
    {
        "type": "folder",
        "path": "data/NKRYA/NKRYA_TEXTS/SCIENCE",
        "tag": "[CTX_SCIENCE]",
        "weight": 50,
        "role": "train",
    },
    {
        "type": "file",
        "path": "data/ustav/ustav_for_training_2.txt",
        "tag": "[CTX_SCIENCE]",
        "weight": 15,
        "role": "train",
    },
]

# ============================================================================
# ОЧИСТКА ТЕКСТА
# ============================================================================

_SPECIAL_RE = re.compile(
    r"(\[CTX_[A-Z_]+\]|\[GAP\]|\[MASK\]|\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\])"
)

_LAT_TO_CYR = {
    "A": "А",
    "a": "а",
    "B": "В",
    "b": "в",
    "E": "Е",
    "e": "е",
    "K": "К",
    "k": "к",
    "M": "М",
    "m": "м",
    "H": "Н",
    "n": "н",
    "O": "О",
    "o": "о",
    "P": "Р",
    "p": "р",
    "C": "С",
    "c": "с",
    "T": "Т",
    "t": "т",
    "y": "у",
    "x": "х",
    "X": "Х",
    "i": "і",
    "I": "І",
}

# р.п. —> руку приложил
# Фл. —> флоринский
# как быть с точками и двоеточиями вокруг букв для обозначения цифр? : г : — пока оставляем так
# учесть специфику символов в грамотах

# в грамотах † +

# почистить эпиграфика левый столбец / правый столбец
# комментарии всё ещё местами (За слогом ба следовал слог га, но последнии затем был зачеркнут киноварью)


# Теперь здесь только системные токены и GAP
SPECIAL_RE = re.compile(r"(<s>|<pad>|</s>|<unk>|<mask>|\[CTX_[A-Z_]+\]|\[GAP\])")

# Канонизация редких символов
_RARE_CHAR_MAP = {
    # punctuation -> "+"
    "†": "+",
    "×": "+",
    # punctuation -> ":"
    "⁘": ":",
    "⁙": ":",
    "⁞": ":",
    "¦": ":",
    # punctuation -> "·"
    "∙": "·",
    "*": "·",
    ".": "·",
    "\uf13f": "·",
    # titlo -> titlo
    "҇": "҃",
    "\uf222": "҃",
    "\uf23a": "҃",
    "\uf2b4": "҃",
    "\uf2b5": "҃",
    "\uf4a5": "҃",
    # rare letters
    "\uf074": "ꙅ",
    "\uf130": "ꙩ",
    "\uf48e": "ꙩ",
    "\uf147": "ѡ",
    "\uf14e": "ѿ",
    "\uf42e": "ѿ",
    "\uf467": "ѯ",
    "\uf47e": "ꙋ",
    "\uf480": "ꙋ",
}

# удаляем явно нежелательные знаки
_DELETE_CHARS = {
    "⃝",
    "⟦",
    "⟧",
    "/",
    "\\",
    "|",
    "?",
    "!",
    '"',
    ";",
    ",",
    "̇",
    "̈",
    "̴",
    "͘",
    "\u200e",  # LRM
    "\uf080",
    "\uf245",
    "\uf265",
    "\uf27a",
    "\uf2db",
    "\uf4a4",
}

_DELETE_RE = re.compile("[" + re.escape("".join(_DELETE_CHARS)) + "]")

# legacy-GAP реликты
_LEGACY_GAP_RE = re.compile(r"___G[АA][РP]___")


def _protect_special_tokens(text: str):
    protected = {}

    def repl(m):
        key = f"QQQ{len(protected)}QQQ"
        protected[key] = m.group(0)
        return key

    return SPECIAL_RE.sub(repl, text), protected


def _unprotect_special_tokens(text: str, protected: dict[str, str]) -> str:
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


def safe_clean_text(line: str) -> str:
    line = line.strip()
    if not line:
        return ""

    m = re.match(r"^(\[CTX_[A-Z_]+\])\s+(.*)", line, re.DOTALL)
    if m:
        tag, text = m.group(1), m.group(2)
    else:
        tag, text = "", line

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"[\ue000-\uf8ff]", "", text)  # остаточный PUA шум
    text = re.sub(r'["\'«»„“”]', "", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)

    # 1) Сначала нормализуем редкие символы грамот/эпиграфики
    for src, dst in _RARE_CHAR_MAP.items():
        text = text.replace(src, dst)

    # 2) Ловим старый реликт GAP до латиницы
    text = _LEGACY_GAP_RE.sub("[GAP]", text)

    # 3) Защищаем спецтокены
    text, protected = _protect_special_tokens(text)

    # 4) Латиница -> кириллица
    for lat, cyr in _LAT_TO_CYR.items():
        text = text.replace(lat, cyr)

    # 5) Удаляем мусорные символы, но оставляем titlo и наши спецзнаки
    text = _DELETE_RE.sub(" ", text)

    # 6) Все прочие знаки препинания убираем, но сохраняем:
    #    буквы, цифры, подчёркивание, пробелы, квадратные скобки, +, :, ·, ҃, (), []
    text = re.sub(r"[^\w\s:\[\]·+҃()]", " ", text)

    # 7) Отделяем спецпунктуацию пробелами
    text = re.sub(r"\s*([:+·])\s*", r" \1 ", text)

    # 8) Возвращаем спецтокены
    text = _unprotect_special_tokens(text, protected)

    # 9) Нормализуем повторяющийся GAP
    text = re.sub(r"(\s*\[GAP\]\s*)+", " [GAP] ", text)

    # 10) Отделяем знаки препинания
    text = re.sub(r"([+:·])([^\s])", r"\1 \2", text)  # отделяем справа
    text = re.sub(r"([^\s])([+:·])", r"\1 \2", text)  # отделяем слева

    # 11) Сжимаем пробелы
    text = re.sub(r"\s+", " ", text).strip()

    return f"{tag} {text}" if tag else text


def has_enough_cyrillic(text: str) -> bool:
    return len(re.findall(r"[а-яА-ЯёЁ\u0400-\u052F\uA640-\uA69F]", text)) >= 3


# ============================================================================
# ЗАГРУЗКА ИСТОЧНИКОВ
# ============================================================================


def iter_raw_lines(cfg: dict):
    """Читает уникальные строки из файла или папки (без тега и веса)."""
    path = PROJECT_ROOT / cfg["path"]
    src_type = cfg["type"]

    if not path.exists():
        print(f"  ⚠️  {path} не найден, пропускаю.")
        return

    if src_type == "file":
        filepaths = [path]
    elif src_type == "folder":
        if not path.is_dir():
            print(f"  ⚠️  {path} не директория, пропускаю.")
            return
        filepaths = sorted([p for p in path.iterdir() if p.suffix == ".txt"])
    else:
        return

    seen = set()
    for fp in filepaths:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    yield line


def load_source(cfg: dict) -> tuple[list[str], list[tuple[str, int]]]:
    """
    Возвращает (train_lines, test_b_raw_lines) для одного источника.

    train_lines   — очищенные строки с тегом, умноженные на weight
    test_b_raw    — список кортежей (сырая строка, weight)
    """
    tag = cfg["tag"]
    weight = cfg["weight"]
    role = cfg.get("role", "train")

    train_lines: list[str] = []
    test_b_raw: list[tuple[str, int]] = []

    for raw in iter_raw_lines(cfg):
        tagged = f"{tag} {raw}".strip()
        if role == "test_b":
            cleaned = safe_clean_text(tagged)
            if cleaned and has_enough_cyrillic(cleaned):
                test_b_raw.append((cleaned, weight))
        else:
            cleaned = safe_clean_text(tagged)
            if cleaned and has_enough_cyrillic(cleaned):
                train_lines.extend([cleaned] * weight)

    return train_lines, test_b_raw


# ============================================================================
# TEST B: обработка скобок
# ============================================================================

ROUND_PAT = re.compile(r"\(([^)]+)\)")
SQUARE_PAT = re.compile(r"\[(?!(?:GAP|MASK|PAD|UNK|CLS|SEP)\]|CTX_)([^\]]+)\]")
CTX_PAT = re.compile(r"^\[CTX_[A-Z_]+\]")


def get_tag(line: str) -> str:
    m = CTX_PAT.match(line.strip())
    return m.group(0) if m else "[CTX_UNKNOWN]"


def is_maskable_test_b_char(ch: str) -> bool:
    """
    Возвращает True только для символов, которые реально стоит маскировать в test_b.
    Пробелы и пунктуация не маскируются.
    """
    if not ch or ch.isspace():
        return False
    cat = unicodedata.category(ch)
    if cat.startswith("P"):  # punctuation
        return False
    return True


def mask_test_b_span(text: str) -> tuple[str, int]:
    """
    Маскирует только 'содержательные' символы внутри span.
    Пробелы и пунктуация сохраняются как есть.

    Returns:
        masked_text, n_masked
    """
    masked_parts = []
    n_masked = 0

    for ch in text:
        if is_maskable_test_b_char(ch):
            masked_parts.append("[MASK]")
            n_masked += 1
        else:
            masked_parts.append(ch)

    return "".join(masked_parts), n_masked


def process_test_b_line(line: str, include_square: bool = True):
    line = line.strip()
    if not line:
        return None

    has_round = bool(ROUND_PAT.search(line))
    has_square = include_square and bool(SQUARE_PAT.search(line))
    if not has_round and not has_square:
        return None

    target = ROUND_PAT.sub(r"\1", line)
    if include_square:
        target = SQUARE_PAT.sub(r"\1", target)

    masked = line
    total_masked = 0
    spans = []

    # Сначала круглые скобки
    def replace_round(m):
        nonlocal total_masked
        inner = m.group(1)
        masked_inner, n_masked = mask_test_b_span(inner)
        total_masked += n_masked
        spans.append({"text": inner, "type": "round"})
        return masked_inner

    masked = ROUND_PAT.sub(replace_round, masked)

    # Потом квадратные скобки, если включены
    if include_square:

        def replace_square(m):
            nonlocal total_masked
            inner = m.group(1)
            masked_inner, n_masked = mask_test_b_span(inner)
            total_masked += n_masked
            spans.append({"text": inner, "type": "square"})
            return masked_inner

        masked = SQUARE_PAT.sub(replace_square, masked)

    if total_masked == 0:
        return None

    return {
        "original": line,
        "masked_input": masked,
        "target": target,
        "tag": get_tag(line),
        "n_masked_chars": total_masked,
        "spans": spans,
    }


# ============================================================================
# РАЗБИЕНИЕ TRAIN -> TRAIN / TEST_A
# ============================================================================


def split_train_lines(
    lines: list[str],
    train_path: Path,
    test_a_path: Path,
    test_ratio: float,
    seed: int,
) -> tuple[int, int]:
    rng = random.Random(seed)
    by_tag = defaultdict(list)

    for line in lines:
        by_tag[get_tag(line)].append(line)

    train_out: list[str] = []
    test_a_out: list[str] = []
    tag_stats = {}

    for tag, tag_lines in sorted(by_tag.items()):
        rng.shuffle(tag_lines)
        n_test = max(1, int(len(tag_lines) * test_ratio))
        test_a_out.extend(tag_lines[:n_test])
        train_out.extend(tag_lines[n_test:])
        tag_stats[tag] = {
            "total": len(tag_lines),
            "train": len(tag_lines) - n_test,
            "test_a": n_test,
        }

    rng.shuffle(train_out)
    rng.shuffle(test_a_out)

    train_path.write_text("\n".join(train_out) + "\n", encoding="utf-8")
    test_a_path.write_text("\n".join(test_a_out) + "\n", encoding="utf-8")

    print(f"\n  {'Тег':<20} {'Всего':>9} {'train':>9} {'test_a':>9}")
    print(f"  {'-' * 52}")
    for tag, s in tag_stats.items():
        print(f"  {tag:<20} {s['total']:>9,} {s['train']:>9,} {s['test_a']:>9,}")

    return len(train_out), len(test_a_out)


# ============================================================================
# MAIN
# ============================================================================


def add_weighted_line(pool: dict[str, int], line: str, weight: int) -> None:
    """Добавляет строку в пул с весом."""
    if line and has_enough_cyrillic(line):
        pool[line] = pool.get(line, 0) + weight


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Формирует train / test_a / test_b из SOURCES_CONFIG"
    )
    parser.add_argument("--out_dir", default="splits")
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_square",
        action="store_true",
        help="Маскировать только (text), игнорировать [text]",
    )
    parser.add_argument(
        "--charters_to_train",
        action="store_true",
        help="Строки test_b БЕЗ скобок —> train-пул с весами",
    )
    parser.add_argument(
        "--test_b_corpus_ratio",
        type=float,
        default=0.5,
        help="Доля test_b, которую добавляем в обычный corpus для train/test_a",
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    include_square = not args.no_square

    print("📂 Загружаем источники...")

    # Весь обычный корпус (train sources + часть test_b)
    all_unique_lines: dict[str, int] = {}  # line -> weight

    # Все test_b строки в очищенном виде
    all_test_b_raw: list[tuple[str, int]] = []

    # ------------------------------------------------------------
    # 1) Сначала читаем ВСЕ источники
    # ------------------------------------------------------------
    for cfg in SOURCES_CONFIG:
        role = cfg.get("role", "train")
        print(f"  {'[test_b]' if role == 'test_b' else '[train] '}  {cfg['path']}")

        for raw in iter_raw_lines(cfg):
            tagged = f"{cfg['tag']} {raw}".strip() if cfg["tag"] else raw
            cleaned = safe_clean_text(tagged)

            if not cleaned or not has_enough_cyrillic(cleaned):
                continue

            if role == "test_b":
                all_test_b_raw.append((cleaned, cfg["weight"]))
            else:
                add_weighted_line(all_unique_lines, cleaned, cfg["weight"])

    print(f"\n  train-пул до test_b: {len(all_unique_lines):,} уникальных строк")
    print(f"  test_b-источники:    {len(all_test_b_raw):,} строк")

    # ------------------------------------------------------------
    # 2) Делим test_b на corpus/eval
    #    corpus -> в train/test_a как обычный текст
    #    eval   -> в test_b.jsonl как real-lacuna
    # ------------------------------------------------------------
    rng = random.Random(args.seed)
    test_b_for_corpus: list[tuple[str, int]] = []
    test_b_for_eval: list[tuple[str, int]] = []

    for raw, weight in all_test_b_raw:
        if rng.random() < args.test_b_corpus_ratio:
            test_b_for_corpus.append((raw, weight))
        else:
            test_b_for_eval.append((raw, weight))

    print(f"  test_b -> corpus: {len(test_b_for_corpus):,}")
    print(f"  test_b -> eval:   {len(test_b_for_eval):,}")

    # ------------------------------------------------------------
    # 3) Добавляем test_b_for_corpus в обычный corpus
    #    ВАЖНО: берём target, если есть скобки
    # ------------------------------------------------------------
    for raw, weight in test_b_for_corpus:
        rec = process_test_b_line(raw, include_square=include_square)
        if rec is None:
            # строка без скобок / без маскируемых spans
            add_weighted_line(all_unique_lines, raw, weight)
        else:
            # скобки убираем, а восстановленный текст идёт в train/test_a
            add_weighted_line(all_unique_lines, rec["target"], weight)

    # ------------------------------------------------------------
    # 4) Если включено, строки без скобок из оставшегося test_b
    #    тоже можно добавить в corpus
    # ------------------------------------------------------------
    no_bracket_count = 0
    if args.charters_to_train:
        for raw, weight in test_b_for_eval:
            rec = process_test_b_line(raw, include_square=include_square)
            if rec is None:
                add_weighted_line(all_unique_lines, raw, weight)
                no_bracket_count += weight
        print(f"  + {no_bracket_count:,} строк без скобок добавлено в train-пул")

    # ------------------------------------------------------------
    # 5) Разбиваем уже ОБЩИЙ корпус на train / test_a
    #    Сплит делаем по УНИКАЛЬНЫМ строкам -> overlap не будет
    # ------------------------------------------------------------
    print("\n📊 Разбиваем на train / test_a (NO OVERLAP)...")

    by_tag = defaultdict(list)
    for line in all_unique_lines.keys():
        by_tag[get_tag(line)].append(line)

    train_unique: list[str] = []
    test_a_unique: list[str] = []
    tag_stats = {}

    rng = random.Random(args.seed)

    for tag, tag_lines in sorted(by_tag.items()):
        rng.shuffle(tag_lines)
        n_test = max(1, int(len(tag_lines) * args.test_ratio))

        test_a_unique.extend(tag_lines[:n_test])
        train_unique.extend(tag_lines[n_test:])

        tag_stats[tag] = {
            "total": len(tag_lines),
            "train": len(tag_lines) - n_test,
            "test_a": n_test,
        }

    rng.shuffle(train_unique)
    rng.shuffle(test_a_unique)

    print(f"\n  {'Тег':<20} {'Всего':>9} {'train':>9} {'test_a':>9}")
    print(f"  {'-' * 52}")
    for tag, s in sorted(tag_stats.items()):
        print(f"  {tag:<20} {s['total']:>9,} {s['train']:>9,} {s['test_a']:>9,}")

    # ------------------------------------------------------------
    # 6) Применяем веса уже ПОСЛЕ split
    # ------------------------------------------------------------
    train_lines_with_weights: list[str] = []
    for line in train_unique:
        weight = all_unique_lines[line]
        train_lines_with_weights.extend([line] * weight)

    # Пишем train (с весами)
    (out / "train.txt").write_text(
        "\n".join(train_lines_with_weights) + "\n",
        encoding="utf-8",
    )

    # test_a — оставляем как уникальные строки (без повторов/весов)
    (out / "test_a.txt").write_text(
        "\n".join(test_a_unique) + "\n",
        encoding="utf-8",
    )

    print("\n✅ train/test_a записаны")
    print(f"  train.txt (с весами): {len(train_lines_with_weights):,}")
    print(f"  test_a.txt (unique):   {len(test_a_unique):,}")
    # ------------------------------------------------------------
    # 7) Строим test_b.jsonl только из test_b_for_eval
    # ------------------------------------------------------------
    print("\n📜 Строим test_b.jsonl...")
    print(f"   Маскировка: (text){'  +  [text]' if include_square else ' только'}")

    records = []
    skipped = 0

    for raw, _weight in test_b_for_eval:
        rec = process_test_b_line(raw, include_square=include_square)
        if rec:
            records.append(rec)
        else:
            skipped += 1

    with (out / "test_b.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"   Записей со скобками: {len(records):,}")
    print(
        f"   Строк без скобок:    {skipped:,}"
        + (" (добавлены в train)" if args.charters_to_train else " (пропущены)")
    )

    print("\n  Примеры записей test_b (первые 3):")
    for i, rec in enumerate(records[:3], start=1):
        print(f"\n  [{i}] original:     {rec['original'][:90]}")
        print(f"       masked_input: {rec['masked_input'][:90]}")
        print(f"       target:       {rec['target'][:90]}")
        print(
            f"       n_masked={rec['n_masked_chars']}  "
            f"spans={[s['text'] for s in rec['spans']]}"
        )

    print("\n" + "=" * 58)
    print("✅ Готово!")
    print(f"   {out}/train.txt     — {len(train_lines_with_weights):,} строк")
    print(f"   {out}/test_a.txt    — {len(test_a_unique):,} строк")
    print(f"   {out}/test_b.jsonl  — {len(records):,} записей")
    print("=" * 58)


if __name__ == "__main__":
    main()
