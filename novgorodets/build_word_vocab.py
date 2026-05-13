import argparse
import json
import re
from collections import Counter
from pathlib import Path

SPECIAL_WORD_TOKENS = [
    "[PAD_WORD]", "[UNK_WORD]",
    "[CLS]", "[SEP]", "[MASK]", "[GAP]",
    "[CTX_CHURCH]", "[CTX_DAILY]", "[CTX_LEGAL]",
    "[CTX_LIT]", "[CTX_EPIC]", "[CTX_SCIENCE]"
]
SPECIAL_RE = re.compile(r"(\[CTX_[A-Z_]+\]|\[GAP\]|\[MASK\]|\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\])")


def iter_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


# пока без нормализации редких символов
def normalize_word_token(tok: str) -> str:
    """
    Нормализация word-level токенов:
    - убираем конечные запятые всегда;
    - убираем конечные точки только у обычных слов;
    - сохраняем внутреннюю пунктуацию и сокращения типа "р.п.".
    """
    # Конечные запятые.
    tok = tok.rstrip(",").rstrip(";")
    # нормализация
    # .rstrip("{"). "}" "(" ")" ! ?
    # учитывать ли регистр?
    if not tok:
        return ""

    # Конечная точка снимается только если это не сокращение с внутренними точками.
    if tok.endswith(".") and "." not in tok[:-1]:
        tok = tok.rstrip(".")

    return tok


def build_vocab(path: Path, vocab_size: int, min_freq: int) -> dict[str, int]:
    counter = Counter()
    for line in iter_lines(path):
        parts = SPECIAL_RE.split(line)
        for part in parts:
            if not part:
                continue
            if SPECIAL_RE.fullmatch(part):
                counter[part] += 1
            else:
                for tok in re.split(r"\s+", part.strip()):
                    if not tok:
                        continue
                    tok = normalize_word_token(tok)
                    if tok:
                        counter[tok] += 1

    tokens = []
    for tok, c in counter.most_common():
        if c < min_freq:
            continue
        tokens.append(tok)

    final_tokens = []
    seen = set()
    for tok in SPECIAL_WORD_TOKENS + tokens:
        if tok not in seen:
            seen.add(tok)
            final_tokens.append(tok)

    if vocab_size > 0:
        final_tokens = final_tokens[:vocab_size]

    return {tok: i for i, tok in enumerate(final_tokens)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="splits/train.txt")
    parser.add_argument("--out_path", default="artifacts/word_vocab.json")
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=2)
    args = parser.parse_args()

    vocab = build_vocab(Path(args.train_path), args.vocab_size, args.min_freq)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved word vocab: {out_path}")
    print(f"Vocab size: {len(vocab)}")


if __name__ == "__main__":
    main()