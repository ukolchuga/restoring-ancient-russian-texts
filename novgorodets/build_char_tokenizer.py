import argparse
import json
import re
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[GAP]",
    "[CTX_CHURCH]", "[CTX_DAILY]", "[CTX_LEGAL]",
    "[CTX_LIT]", "[CTX_EPIC]", "[CTX_SCIENCE]",
    "+", ":", "·"
]
SPECIAL_RE = re.compile(
    r"(\[CTX_[A-Z_]+\]|\[GAP\]|\[MASK\]|\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\]|[+:·])"
)

def iter_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def collect_chars(path: Path, min_freq: int) -> list[str]:
    counter = Counter()
    for line in iter_lines(path):
        parts = SPECIAL_RE.split(line)
        for part in parts:
            if not part:
                continue
            if SPECIAL_RE.fullmatch(part):
                continue
            for ch in part:
                counter[ch] += 1
    chars = [ch for ch, c in counter.items() if c >= min_freq]
    chars.sort()
    return chars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="splits/train.txt")
    parser.add_argument("--out_dir", default="artifacts/char_tokenizer")
    parser.add_argument("--min_freq", type=int, default=1)
    args = parser.parse_args()

    train_path = Path(args.train_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chars = collect_chars(train_path, args.min_freq)
    vocab_tokens = []
    seen = set()

    for tok in SPECIAL_TOKENS + chars:
        if tok not in seen:
            seen.add(tok)
            vocab_tokens.append(tok)

    vocab = {tok: i for i, tok in enumerate(vocab_tokens)}
    cfg = {
        "special_tokens": SPECIAL_TOKENS,
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
    }

    (out_dir / "char_vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved char vocab: {out_dir / 'char_vocab.json'}")
    print(f"Vocab size: {len(vocab)}")


if __name__ == "__main__":
    main()