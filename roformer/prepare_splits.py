#!/usr/bin/env python3
"""
prepare_splits.py for RoFormer V2 Pipeline.
Adapted from the novgorodets pipeline to work independently.

Reads raw sources, applies tags/weights, cleans text, and splits into:
  - splits/train.txt
  - splits/test_a.txt
  - splits/test_b.jsonl
"""

import argparse
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

# Set project root relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# SOURCES CONFIGURATION
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
# TEXT CLEANING
# ============================================================================

_LAT_TO_CYR = {
    "a": "а",
    "b": "в",
    "e": "е",
    "k": "к",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "р",
    "c": "с",
    "t": "т",
    "y": "у",
    "x": "х",
    "i": "і",
}


SPECIAL_RE = re.compile(r"(<s>|<pad>|</s>|<unk>|<mask>|\[CTX_[A-Z_]+\]|\[GAP\])")

_RARE_CHAR_MAP = {
    "†": "+",
    "×": "+",
    "⁘": ":",
    "⁙": ":",
    "⁞": ":",
    "¦": ":",
    "∙": "·",
    "*": "·",
    ".": "·",
    "\uf13f": "·",
    "҇": "҃",
    "\uf222": "҃",
    "\uf23a": "҃",
    "\uf2b4": "҃",
    "\uf2b5": "҃",
    "\uf4a5": "҃",
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
    "\u200e",
    "\uf080",
    "\uf245",
    "\uf265",
    "\uf27a",
    "\uf2db",
    "\uf4a4",
}
_DELETE_RE = re.compile("[" + re.escape("".join(_DELETE_CHARS)) + "]")
_LEGACY_GAP_RE = re.compile(r"___G[АA][РP]___")

CYR_NUMERALS = "авгдєѕзиѳіклмнѯопрстуфхѱѡцчшщъыьѣюѧѩѯѱѳѵ"
NUM_PATTERN = re.compile(rf"([:+·])([{CYR_NUMERALS}]+҃)\1")


def _protect_numerals(text: str):
    protected_nums = {}

    def repl(m):
        key = f"PNUM{len(protected_nums)}PNUM"
        protected_nums[key] = m.group(0)
        return key

    # Защищаем конструкции типа :л҃: или ·в·
    return NUM_PATTERN.sub(repl, text), protected_nums


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
    tag, text = (m.group(1), m.group(2)) if m else ("", line)

    text = text.lower()
    text = re.sub(r"___g[аa][рp]___|\[gap\]", "[GAP]", text)

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "").replace("\u200b", "")

    text = re.sub(r"[\ue000-\uf8ff]", "", text)
    text = re.sub(r'["\'«»„“”]', "", text)

    text = re.sub(r"[\u0300-\u036f]|[\u0484-\u0489]", "", text)

    for src, dst in _RARE_CHAR_MAP.items():
        text = text.replace(src, dst)

    text = _LEGACY_GAP_RE.sub("[GAP]", text)

    text, protected = _protect_special_tokens(text)
    text, protected_nums = _protect_numerals(text)

    for lat, cyr in _LAT_TO_CYR.items():
        text = text.replace(lat, cyr)

    text = _DELETE_RE.sub(" ", text)
    text = re.sub(r"[^\w\s:\[\]·+҃()]", " ", text)
    text = re.sub(r"\s*([:+·])\s*", r" \1 ", text)
    for key, val in protected_nums.items():
        text = text.replace(key, f" {val} ")

    text = _unprotect_special_tokens(text, protected)

    text = re.sub(r"(\s*\[GAP\]\s*)+", " [GAP] ", text)
    text = re.sub(r"([+:·])([^\s])", r"\1 \2", text)
    text = re.sub(r"([^\s])([+:·])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()

    return f"{tag} {text}" if tag else text


def has_enough_cyrillic(text: str) -> bool:
    return len(re.findall(r"[а-яА-ЯёЁ\u0400-\u052F\uA640-\uA69F]", text)) >= 3


# ============================================================================
# SOURCE LOADING
# ============================================================================


def iter_raw_lines(cfg: dict):
    path = PROJECT_ROOT / cfg["path"]
    src_type = cfg["type"]
    if not path.exists():
        return
    filepaths = (
        [path]
        if src_type == "file"
        else sorted([p for p in path.iterdir() if p.suffix == ".txt"])
    )
    seen = set()
    for fp in filepaths:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    yield line


def load_source(cfg: dict):
    tag, weight, role = cfg["tag"], cfg["weight"], cfg.get("role", "train")
    train_lines, test_b_raw = [], []
    for raw in iter_raw_lines(cfg):
        tagged = f"{tag} {raw}".strip()
        cleaned = safe_clean_text(tagged)
        if cleaned and has_enough_cyrillic(cleaned):
            if role == "test_b":
                test_b_raw.append((cleaned, weight))
            else:
                train_lines.extend([cleaned] * weight)
    return train_lines, test_b_raw


# ============================================================================
# SPLITTING & SAVING
# ============================================================================

ROUND_PAT = re.compile(r"\(([^)]+)\)")
SQUARE_PAT = re.compile(r"\[(?!(?:GAP|MASK|PAD|UNK|CLS|SEP)\]|CTX_)([^\]]+)\]")
CTX_PAT = re.compile(r"^\[CTX_[A-Z_]+\]")


def get_tag(line: str) -> str:
    m = CTX_PAT.match(line.strip())
    return m.group(0) if m else "[CTX_UNKNOWN]"


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

    return {
        "original": line,
        "target": target,
        "tag": get_tag(line),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="splits")
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_square", action="store_true")
    parser.add_argument("--charters_to_train", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_train, all_test_b_raw = [], []
    for cfg in SOURCES_CONFIG:
        tr, tb = load_source(cfg)
        all_train.extend(tr)
        all_test_b_raw.extend(tb)

    if args.charters_to_train:
        for raw, w in all_test_b_raw:
            if process_test_b_line(raw, not args.no_square) is None:
                c = safe_clean_text(raw)
                if c and has_enough_cyrillic(c):
                    all_train.extend([c] * w)

    rng = random.Random(args.seed)
    rng.shuffle(all_train)
    by_tag = defaultdict(list)
    for l in all_train:
        by_tag[get_tag(l)].append(l)

    train_final, test_a_final = [], []
    for tag, lines in by_tag.items():
        rng.shuffle(lines)
        n_test = max(1, int(len(lines) * args.test_ratio))
        test_a_final.extend(lines[:n_test])
        train_final.extend(lines[n_test:])

    rng.shuffle(train_final)
    rng.shuffle(test_a_final)
    (out / "train.txt").write_text("\n".join(train_final) + "\n", encoding="utf-8")
    (out / "test_a.txt").write_text("\n".join(test_a_final) + "\n", encoding="utf-8")

    records = [process_test_b_line(r, not args.no_square) for r, _ in all_test_b_raw]
    records = [r for r in records if r]
    with (out / "test_b.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Created splits in {args.out_dir}:")
    print(f"   train.txt: {len(train_final):,} lines")
    print(f"   test_a.txt: {len(test_a_final):,} lines")
    print(f"   test_b.jsonl: {len(records):,} records")


if __name__ == "__main__":
    main()
