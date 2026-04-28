import json
import re
from pathlib import Path

SPECIAL_RE = re.compile(
    r"(\[CTX_[A-Z_]+\]|\[GAP\]|\[MASK\]|\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\]|[+:·])"
)


def load_vocab(path: str | Path) -> dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_special(text: str) -> list[str]:
    return [p for p in SPECIAL_RE.split(text) if p]


def align_char_to_word(
    text: str,
    char_vocab: dict[str, int],
    word_vocab: dict[str, int],
    max_len: int = 256,
    add_cls_sep: bool = True,
):
    char_unk = char_vocab["[UNK]"]
    char_pad = char_vocab["[PAD]"]
    char_cls = char_vocab["[CLS]"]
    char_sep = char_vocab["[SEP]"]

    word_unk = word_vocab["[UNK_WORD]"]
    word_pad = word_vocab["[PAD_WORD]"]

    special_char_ids = {char_vocab[t] for t in char_vocab if t.startswith("[") and t.endswith("]")}
    input_ids = []
    word_ids = []

    if add_cls_sep:
        input_ids.append(char_cls)
        word_ids.append(word_vocab.get("[CLS]", word_unk))

    for part in split_special(text.strip()):
        if SPECIAL_RE.fullmatch(part):
            input_ids.append(char_vocab.get(part, char_unk))
            word_ids.append(word_vocab.get(part, word_unk))
            continue

        chunks = re.split(r"(\s+)", part)
        for chunk in chunks:
            if not chunk:
                continue
            if chunk.isspace():
                for ch in chunk:
                    input_ids.append(char_vocab.get(ch, char_unk))
                    word_ids.append(word_unk)
            else:
                wid = word_vocab.get(chunk, word_unk)
                for ch in chunk:
                    input_ids.append(char_vocab.get(ch, char_unk))
                    word_ids.append(wid)

    if add_cls_sep:
        input_ids.append(char_sep)
        word_ids.append(word_vocab.get("[SEP]", word_unk))

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        word_ids = word_ids[:max_len]
        if add_cls_sep:
            input_ids[-1] = char_sep
            word_ids[-1] = word_vocab.get("[SEP]", word_unk)

    attention_mask = [1] * len(input_ids)
    special_tokens_mask = [1 if tid in special_char_ids else 0 for tid in input_ids]

    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids.extend([char_pad] * pad_len)
        word_ids.extend([word_pad] * pad_len)
        attention_mask.extend([0] * pad_len)
        special_tokens_mask.extend([1] * pad_len)

    return {
        "input_ids": input_ids,
        "word_ids": word_ids,
        "attention_mask": attention_mask,
        "special_tokens_mask": special_tokens_mask,
    }