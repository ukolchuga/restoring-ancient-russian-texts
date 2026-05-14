#!/usr/bin/env python3
"""
prepare_splits.py

Читает исходные файлы по SOURCES_CONFIG, применяет теги и веса,
очищает текст, затем делит данные на три части:

  splits/train.txt     — основной корпус (90%), стратифицировано по CTX-тегу
  splits/eval.txt      — основной корпус (5%), валидация
  splits/test_a.txt    — основной корпус (5%), искусственные пропуски генерирует коллатор
  splits/test_b.jsonl  — источники со скобками; (text) и [text] → [MASK]×len

Поле "role" в SOURCES_CONFIG:
  "train"  — строки идут в train/eval/test_a (веса применяются)
  "test_b" — строки СО скобками → test_b.jsonl
             строки БЕЗ скобок  → corpus (train/eval/test_a, веса применяются)

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
  python prepare_splits.py --out_dir splits_tags
  python prepare_splits.py --out_dir splits_tags --no_square
"""

import os
import re
import random
import unicodedata
import json
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCES_CONFIG = [
    # -- Birchbark & epigraphica (test_b) ----------------------------------------------------
    {"type": "file", "path": "cleaned_data/birchbark/birchbark_DAILY.txt", "tag": "[CTX_DAILY]", "weight": 40, "role": "test_b"},
    {"type": "file", "path": "cleaned_data/birchbark/birchbark_CHURCH.txt", "tag": "[CTX_CHURCH]", "weight": 40, "role": "test_b"},
    {"type": "file", "path": "cleaned_data/birchbark/birchbark_LIT.txt", "tag": "[CTX_LIT]", "weight": 40, "role": "test_b"},
    {"type": "file", "path": "cleaned_data/birchbark/birchbark_LEGAL.txt", "tag": "[CTX_LEGAL]", "weight": 40, "role": "test_b"},
    {"type": "file", "path": "cleaned_data/epigraphica/epigraphica_final_cleaned_with_brackets.txt", "tag": "[CTX_CHURCH]", "weight": 20, "role": "test_b"},

    # -- DAILY ----------------------------------------------------------------------------------
    {"type": "file", "path": "cleaned_data/diacu/diacu_DAILY.txt", "tag": "[CTX_DAILY]", "weight": 34, "role": "train"},
    {"type": "file", "path": "cleaned_data/torot/torot_DAILY.txt", "tag": "[CTX_DAILY]", "weight": 34, "role": "train"},
    {"type": "file", "path": "cleaned_data/pushkinskij_texts/pushkinskij_DAILY.txt", "tag": "[CTX_DAILY]", "weight": 34, "role": "train"},
    {"type": "file", "path": "cleaned_data/nkrya/nkrya_DAILY.txt", "tag": "[CTX_DAILY]", "weight": 34, "role": "train"},

    # -- LEGAL -----------------------------------------------------------------------------------
    {"type": "file", "path": "cleaned_data/diacu/diacu_LEGAL.txt", "tag": "[CTX_LEGAL]", "weight": 6, "role": "train"},
    {"type": "file", "path": "cleaned_data/torot/torot_LEGAL.txt", "tag": "[CTX_LEGAL]", "weight": 10, "role": "train"},
    {"type": "file", "path": "cleaned_data/pushkinskij_texts/pushkinskij_LEGAL.txt", "tag": "[CTX_LEGAL]", "weight": 7, "role": "train"},
    {"type": "file", "path": "cleaned_data/nkrya/nkrya_LEGAL.txt", "tag": "[CTX_LEGAL]", "weight": 7, "role": "train"},
    {"type": "file", "path": "cleaned_data/nkrya/OND_LEGAL.txt", "tag": "[CTX_LEGAL]", "weight": 10, "role": "train"},

    # -- CHURCH ------------------------------------------------------------------------------------
    {"type": "file", "path": "cleaned_data/bible/bible_final_cleaned.txt", "tag": "[CTX_CHURCH]", "weight": 1, "role": "train"},
    {"type": "file", "path": "cleaned_data/diacu/diacu_CHURCH.txt", "tag": "[CTX_CHURCH]", "weight": 1, "role": "train"},
    {"type": "file", "path": "cleaned_data/torot/torot_CHURCH.txt", "tag": "[CTX_CHURCH]", "weight": 2, "role": "train"},
    {"type": "file", "path": "cleaned_data/pushkinskij_texts/pushkinskij_CHURCH.txt", "tag": "[CTX_CHURCH]", "weight": 5, "role": "train"},

    # -- LIT --------------------------------------------------------------------------------------
    {"type": "file", "path": "cleaned_data/diacu/diacu_LIT.txt", "tag": "[CTX_LIT]", "weight": 3, "role": "train"},
    {"type": "file", "path": "cleaned_data/torot/torot_LIT.txt", "tag": "[CTX_LIT]", "weight": 3, "role": "train"},
    {"type": "file", "path": "cleaned_data/pushkinskij_texts/pushkinskij_LIT.txt", "tag": "[CTX_LIT]", "weight": 3, "role": "train"},
    {"type": "file", "path": "cleaned_data/nkrya/nkrya_LIT.txt", "tag": "[CTX_LIT]", "weight": 3, "role": "train"},
    {"type": "file", "path": "cleaned_data/nkrya/OND_LIT.txt", "tag": "[CTX_LIT]", "weight": 3, "role": "train"},

    # -- EPIC --------------------------------------------------------------------------------------
    {"type": "file", "path": "cleaned_data/bylini/clean_original_novoya_zapis.txt", "tag": "[CTX_EPIC]", "weight": 6, "role": "train"},
    {"type": "file", "path": "cleaned_data/bylini/clean_original_staraya_zapis.txt", "tag": "[CTX_EPIC]", "weight": 6, "role": "train"},
    {"type": "file", "path": "cleaned_data/pushkinskij_texts/pushkinskij_EPIC.txt", "tag": "[CTX_EPIC]", "weight": 6, "role": "train"},

    # -- SCIENCE -------------------------------------------------------------------------------------
    {"type": "file", "path": "cleaned_data/pushkinskij_texts/pushkinskij_SCIENCE.txt", "tag": "[CTX_SCIENCE]", "weight": 14, "role": "train"},
    {"type": "file", "path": "cleaned_data/nkrya/nkrya_SCIENCE.txt", "tag": "[CTX_SCIENCE]", "weight": 14, "role": "train"},
    {"type": "file", "path": "cleaned_data/ustav/ustav_final_cleaned.txt", "tag": "[CTX_SCIENCE]", "weight": 14, "role": "train"},
]


# ============================================================================
# ОЧИСТКА ТЕКСТА
# ============================================================================

_SPECIAL_RE = re.compile(
    r"(\[CTX_[A-Z_]+\]|\[GAP\]|\[MASK\]|\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\])"
)

_LAT_TO_CYR = {
    "A": "А", "a": "а", "B": "В", "b": "в", "E": "Е", "e": "е",
    "K": "К", "k": "к", "M": "М", "m": "м", "H": "Н", "n": "н",
    "O": "О", "o": "о", "P": "Р", "p": "р", "C": "С", "c": "с",
    "T": "Т", "t": "т", "y": "у", "x": "х", "X": "Х",
    "i": "і", "I": "І", "w": "ѡ", "W": "ѡ",
}

SPECIAL_RE = re.compile(r"(<s>|<pad>|</s>|<unk>|<mask>|\[CTX_[A-Z_]+\]|\[GAP\])")

_RARE_CHAR_MAP = {
    "†": "+", "×": "+", "⁘": ":", "⁙": ":", "⁞": ":", "¦": ":",
    "∙": "·", "*": "·", ".": "·", "\uf13f": "·",
    "҇": "҃", "\uf222": "҃", "\uf23a": "҃", "\uf2b4": "҃", "\uf2b5": "҃", "\uf4a5": "҃", "ᵕ": "҃",
    "\uf074": "ѕ", "ᴤ": "ѕ", "ꙅ": "ѕ",
    "\uf130": "ꙩ", "\uf48e": "ꙩ", "ꙫ": "ꙩ", "ꙭ": "ꙩ",
    "ӧ": "о", "ᲂ": "о", "ꛩ": "о", "ѻ": "о",
    "\uf147": "ѡ", "ꙍ": "ѡ",
    "\uf14e": "ѿ", "\uf42e": "ѿ", "ὼ": "ѿ", "ѽ": "ѿ", "Ѿ": "ѿ",
    "\uf467": "ѯ",
    "\uf47e": "ꙋ", "\uf480": "ꙋ", "ȣ": "ꙋ",
    "ȥ": "ꙁ", "ꙥ": "л", "ꙇ": "і", "ⱚ": "ѳ", "ꙛ": "ѫ", "ꙙ": "ѧ",
    "ҍ": "ѣ", "ⱕ": "є", "ⱔ": "є",
    "ⰴ": "д", "ⱉ": "ѿ", "ꙕ": "оі", "ⱖ": "ѭ", "ⰹ": "ï", "ꙉ": "г",
    "ḯ": "ï", "ѷ": "ѵ",
    "ӓ": "а", "ӱ": "у", "ӹ": "ы", "ӥ": "и",
}

_DELETE_CHARS = {
    "⃝", "⟦", "⟧", "/", "\\", "|", "?", "!", '"', ";", ",",
    "̇", "̈", "̴", "͘", "ⸯ", "\u200e", "ꙿ", "ʹ", "ʼ", "ҁ",
    "\uf080", "\uf245", "\uf265", "\uf27a", "\uf2db", "\uf4a4",
}

_DELETE_RE = re.compile("[" + re.escape("".join(_DELETE_CHARS)) + "]")
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
    text = re.sub(r"[\ue000-\uf8ff]", "", text)
    text = re.sub(r'["\'«»„""]', "", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)

    for src, dst in _RARE_CHAR_MAP.items():
        text = text.replace(src, dst)

    text = _LEGACY_GAP_RE.sub("[GAP]", text)
    text, protected = _protect_special_tokens(text)

    for lat, cyr in _LAT_TO_CYR.items():
        text = text.replace(lat, cyr)

    text = _DELETE_RE.sub(" ", text)
    text = re.sub(r"[^\w\s:\[\]·+҃()]", " ", text)
    text = re.sub(r"\s*([:+·])\s*", r" \1 ", text)
    text = _unprotect_special_tokens(text, protected)
    text = re.sub(r"(\s*\[GAP\]\s*)+", " [GAP] ", text)
    text = re.sub(r"([+:·])([^\s])", r"\1 \2", text)
    text = re.sub(r"([^\s])([+:·])", r"\1 \2", text)
    text = re.sub(r"([\(\[])\s*([+:·])", r"\1\2", text)
    text = re.sub(r"([+:·])\s*([\)\]])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()

    return f"{tag} {text}" if tag else text


def has_enough_cyrillic(text: str) -> bool:
    return len(re.findall(r"[а-яА-ЯёЁ\u0400-\u052F\uA640-\uA69F]", text)) >= 3


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


# ============================================================================
# ЗАГРУЗКА ИСТОЧНИКОВ
# ============================================================================

def iter_raw_lines(cfg: dict):
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


# ============================================================================
# TEST B: обработка скобок
# ============================================================================

ROUND_PAT  = re.compile(r"\(([^)]+)\)")
SQUARE_PAT = re.compile(r"\[(?!(?:GAP|MASK|PAD|UNK|CLS|SEP)\]|CTX_)([^\]]+)\]")
CTX_PAT    = re.compile(r"^\[CTX_[A-Z_]+\]")


def get_tag(line: str) -> str:
    m = CTX_PAT.match(line.strip())
    return m.group(0) if m else "[CTX_UNKNOWN]"


def has_brackets(line: str, include_square: bool) -> bool:
    if ROUND_PAT.search(line):
        return True
    if include_square and SQUARE_PAT.search(line):
        return True
    return False


def mask_test_b_span(text: str) -> tuple[str, int]:
    return "[MASK]" * len(text), len(text)


def process_test_b_line(line: str, include_square: bool = True):
    line = line.strip()
    if not line:
        return None

    has_round  = bool(ROUND_PAT.search(line))
    has_square = include_square and bool(SQUARE_PAT.search(line))
    if not has_round and not has_square:
        return None

    target = ROUND_PAT.sub(r"\1", line)
    if include_square:
        target = SQUARE_PAT.sub(r"\1", target)

    masked = line
    total_masked = 0
    spans = []

    def replace_round(m):
        nonlocal total_masked
        inner = m.group(1)
        masked_inner, n_masked = mask_test_b_span(inner)
        total_masked += n_masked
        spans.append({"text": inner, "type": "round"})
        return masked_inner

    masked = ROUND_PAT.sub(replace_round, masked)

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
        "original":       line,
        "masked_input":   masked,
        "target":         target,
        "tag":            get_tag(line),
        "n_masked_chars": total_masked,
        "spans":          spans,
    }


# ============================================================================
# MAIN
# ============================================================================

def add_weighted_line(pool: dict[str, int], line: str, weight: int) -> None:
    if line and has_enough_cyrillic(line):
        pool[line] = pool.get(line, 0) + weight


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Формирует train / eval / test_a / test_b из SOURCES_CONFIG"
    )
    parser.add_argument("--out_dir",    default="splits_tags")
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument(
        "--no_square",
        action="store_true",
        help="Маскировать только (text), игнорировать [text]",
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    include_square = not args.no_square

    print("📂 Загружаем источники...")

    all_unique_lines: dict[str, int] = {}
    all_test_b_bracket: list[str]    = []   # строки СО скобками → test_b.jsonl

    source_word_stats:   dict[str, int] = defaultdict(int)
    category_word_stats: dict[str, int] = defaultdict(int)

    # -------- 1) Читаем ВСЕ источники --------
    for cfg in SOURCES_CONFIG:
        role = cfg.get("role", "train")
        print(f"  {'[test_b]' if role == 'test_b' else '[train] '}  {cfg['path']}")

        for raw in iter_raw_lines(cfg):
            tagged  = f"{cfg['tag']} {raw}".strip() if cfg["tag"] else raw
            cleaned = safe_clean_text(tagged)

            if not cleaned or not has_enough_cyrillic(cleaned):
                continue

            word_count = count_words(cleaned)
            source_word_stats[cfg["path"]] += word_count
            category_word_stats[cfg["tag"] or "[NO_TAG]"] += word_count

            if role == "test_b":
                # Детерминированная маршрутизация по наличию скобок
                if has_brackets(cleaned, include_square):
                    all_test_b_bracket.append(cleaned)
                else:
                    # Без скобок — в corpus с весом
                    add_weighted_line(all_unique_lines, cleaned, cfg["weight"])
            else:
                add_weighted_line(all_unique_lines, cleaned, cfg["weight"])

    print(f"\n  train-пул:    {len(all_unique_lines):,} уникальных строк")
    print(f"  test_b (со скобками): {len(all_test_b_bracket):,} строк")

    # -------- 2) Разбиваем corpus на train / eval / test_a --------
    print("\n📊 Разбиваем на train / eval / test_a (NO OVERLAP)...")

    by_tag = defaultdict(list)
    for line in all_unique_lines.keys():
        by_tag[get_tag(line)].append(line)

    train_unique:  list[str] = []
    eval_unique:   list[str] = []
    test_a_unique: list[str] = []
    tag_stats = {}

    rng = random.Random(args.seed)

    def split_counts(n: int, eval_ratio: float, test_ratio: float) -> tuple[int, int, int]:
        if n <= 0: return 0, 0, 0
        if n == 1: return 1, 0, 0
        if n == 2: return 1, 1, 0

        n_eval = int(round(n * eval_ratio))
        n_test = int(round(n * test_ratio))

        if n_eval == 0: n_eval = 1
        if n_test == 0: n_test = 1

        while n_eval + n_test >= n:
            if n_eval > n_test and n_eval > 1: n_eval -= 1
            elif n_test > 1:                   n_test -= 1
            else:                              break

        n_train = n - n_eval - n_test
        if n_train < 1:
            n_train = 1
            if n_eval > n_test and n_eval > 1: n_eval -= 1
            elif n_test > 1:                   n_test -= 1

        return n_train, n_eval, n_test

    for tag, tag_lines in sorted(by_tag.items()):
        rng.shuffle(tag_lines)
        n_total = len(tag_lines)
        n_train, n_eval, n_test = split_counts(n_total, args.eval_ratio, args.test_ratio)

        eval_unique.extend(tag_lines[:n_eval])
        test_a_unique.extend(tag_lines[n_eval:n_eval + n_test])
        train_unique.extend(tag_lines[n_eval + n_test:])

        tag_stats[tag] = {"total": n_total, "train": n_train, "eval": n_eval, "test_a": n_test}

    rng.shuffle(train_unique)
    rng.shuffle(eval_unique)
    rng.shuffle(test_a_unique)

    print(f"\n  {'Тег':<20} {'Всего':>9} {'train':>9} {'eval':>9} {'test_a':>9}")
    print(f"  {'-' * 63}")
    for tag, s in sorted(tag_stats.items()):
        print(f"  {tag:<20} {s['total']:>9,} {s['train']:>9,} {s['eval']:>9,} {s['test_a']:>9,}")

    # -------- 3) Применяем веса ПОСЛЕ split --------
    train_lines_with_weights: list[str] = []
    for line in train_unique:
        weight = all_unique_lines[line]
        train_lines_with_weights.extend([line] * weight)

    (out / "train.txt").write_text("\n".join(train_lines_with_weights) + "\n", encoding="utf-8")
    (out / "eval.txt").write_text("\n".join(eval_unique) + "\n", encoding="utf-8")
    (out / "test_a.txt").write_text("\n".join(test_a_unique) + "\n", encoding="utf-8")

    print("\n✅ train/eval/test_a записаны")
    print(f"  train.txt (с весами): {len(train_lines_with_weights):,}")
    print(f"  eval.txt (unique):    {len(eval_unique):,}")
    print(f"  test_a.txt (unique):  {len(test_a_unique):,}")

    # -------- 4) Строим test_b.jsonl --------
    print("\n📜 Строим test_b.jsonl...")
    print(f"   Маскировка: (text){'  +  [text]' if include_square else ' только'}")

    records = []
    for raw in all_test_b_bracket:
        rec = process_test_b_line(raw, include_square=include_square)
        if rec:
            records.append(rec)

    with (out / "test_b.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"   Записей: {len(records):,}")

    # -------- 5) Подсчёт слов --------
    train_word_count  = sum(count_words(l) for l in train_lines_with_weights)
    eval_word_count   = sum(count_words(l) for l in eval_unique)
    test_a_word_count = sum(count_words(l) for l in test_a_unique)
    test_b_word_count = sum(count_words(r["masked_input"]) for r in records)
    test_b_target_wc  = sum(count_words(r["target"])       for r in records)
    test_b_masked_ch  = sum(r["n_masked_chars"]            for r in records)
    test_b_target_ch  = sum(len(s["text"]) for r in records for s in r.get("spans", []))
    total_final       = train_word_count + eval_word_count + test_a_word_count + test_b_target_wc

    print("\n📊 Подсчёт слов по источникам:")
    for cfg in SOURCES_CONFIG:
        print(f"  {cfg['path']:<70} {source_word_stats.get(cfg['path'], 0):>10,}")

    print("\n📊 Подсчёт слов по категориям:")
    for tag in sorted(category_word_stats):
        print(f"  {tag:<15} {category_word_stats[tag]:>10,}")

    print("\n📊 Подсчёт слов в итоговых датасетах:")
    print(f"  train.txt        {train_word_count:>10,}")
    print(f"  eval.txt         {eval_word_count:>10,}")
    print(f"  test_a.txt       {test_a_word_count:>10,}")
    print(f"  test_b.jsonl     {test_b_word_count:>10,}  (masked_input)")
    print(f"  test_b.jsonl     {test_b_target_wc:>10,}  (target)")
    print(f"  test_b.jsonl     {test_b_masked_ch:>10,}  (masked_chars)")
    print(f"  test_b.jsonl     {test_b_target_ch:>10,}  (target_chars)")
    print(f"  total            {total_final:>10,}")


if __name__ == "__main__":
    main()