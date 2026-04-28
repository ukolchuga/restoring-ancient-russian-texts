import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from align_dual import load_vocab, align_char_to_word


def read_txt_lines(path: Path, limit: int = 0):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
                if limit and len(out) >= limit:
                    break
    return out


def read_jsonl(path: Path, limit: int = 0):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def encode_lines(lines, char_vocab, word_vocab, max_len):
    return [
        align_char_to_word(text, char_vocab, word_vocab, max_len=max_len, add_cls_sep=True)
        for text in lines
    ]


def encode_test_b(records, char_vocab, word_vocab, max_len):
    mask_id = char_vocab["[MASK]"]
    out = []
    for rec in records:
        enc_in = align_char_to_word(rec["masked_input"], char_vocab, word_vocab, max_len=max_len, add_cls_sep=True)
        enc_tg = align_char_to_word(rec["target"], char_vocab, word_vocab, max_len=max_len, add_cls_sep=True)

        labels = [-100] * max_len
        for i, (inp, tgt, am) in enumerate(zip(enc_in["input_ids"], enc_tg["input_ids"], enc_in["attention_mask"])):
            if am == 1 and inp == mask_id:
                labels[i] = tgt

        ex = dict(enc_in)
        ex["labels"] = labels
        ex["original"] = rec.get("original", "")
        ex["target_text"] = rec.get("target", "")
        out.append(ex)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="splits/train.txt")
    parser.add_argument("--test_a_path", default="splits/test_a.txt")
    parser.add_argument("--test_b_path", default="splits/test_b.jsonl")
    parser.add_argument("--char_vocab_path", default="artifacts/char_tokenizer/char_vocab.json")
    parser.add_argument("--word_vocab_path", default="artifacts/word_vocab.json")
    parser.add_argument("--out_dir", default="artifacts/dual_dataset")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Debug mode: first N rows per split")
    args = parser.parse_args()

    char_vocab = load_vocab(args.char_vocab_path)
    word_vocab = load_vocab(args.word_vocab_path)

    train_lines = read_txt_lines(Path(args.train_path), limit=args.limit)
    test_a_lines = read_txt_lines(Path(args.test_a_path), limit=args.limit)
    test_b_records = read_jsonl(Path(args.test_b_path), limit=args.limit)

    train_ds = Dataset.from_list(encode_lines(train_lines, char_vocab, word_vocab, args.max_len))
    test_a_ds = Dataset.from_list(encode_lines(test_a_lines, char_vocab, word_vocab, args.max_len))
    test_b_ds = Dataset.from_list(encode_test_b(test_b_records, char_vocab, word_vocab, args.max_len))

    ds = DatasetDict({"train": train_ds, "test_a": test_a_ds, "test_b": test_b_ds})
    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))

    print(f"Saved dataset: {out_dir}")
    print(ds)


if __name__ == "__main__":
    main()
