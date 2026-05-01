import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizerFast


def read_txt_lines(path: Path, limit: int = 0):
    """Reads lines from a text file, skipping empty lines."""
    out = []
    if not path.exists():
        print(f"Warning: {path} not found.")
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
                if limit and len(out) >= limit:
                    break
    return out


def read_jsonl(path: Path, limit: int = 0):
    """Reads records from a JSONL file."""
    out = []
    if not path.exists():
        print(f"Warning: {path} not found.")
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def encode_lines(lines, tokenizer):
    """Tokenizes lines without truncation/padding to allow for later block packing."""
    return tokenizer(
        lines,
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )


def group_texts(examples, block_size):
    """
    Packs tokens into fixed-size blocks to maximize training efficiency.
    Unused remainders are discarded.
    """
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def encode_test_b(records, tokenizer, max_len):
    """
    Prepares the fixed evaluation set (test_b).
    Maps character-level masks from the source to BPE tokens.
    """
    out = []
    for rec in records:
        original_text = rec.get("original", "")
        if not original_text:
            continue

        # Encode input (with masks) and target separately
        enc_in = tokenizer(
            rec["masked_input"],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        enc_tg = tokenizer(
            rec["target"], truncation=True, max_length=max_len, padding="max_length"
        )

        ids_in = enc_in["input_ids"]
        ids_tg = enc_tg["input_ids"]
        mask_id = tokenizer.mask_token_id

        # Align mask positions to target tokens
        labels = [-100] * max_len
        if len(ids_in) == len(ids_tg):
            for i in range(len(ids_in)):
                if ids_in[i] == mask_id:
                    labels[i] = ids_tg[i]

        out.append({
            "input_ids": ids_in,
            "attention_mask": enc_in["attention_mask"],
            "labels": labels,
            "original": original_text,
            "target_text": rec.get("target", ""),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare RoFormer datasets with token packing.")
    parser.add_argument("--splits_dir", default="splits", help="Directory containing split files.")
    parser.add_argument("--tokenizer_path", default="ancient_rus_tokenizer_BPE", help="Path to the BPE tokenizer.")
    parser.add_argument("--out_dir", default="artifacts/roformer_dataset", help="Output directory for processed datasets.")
    parser.add_argument("--max_len", type=int, default=256, help="Block size for token packing.")
    parser.add_argument("--limit", type=int, default=0, help="Debug mode: limit number of samples.")
    args = parser.parse_args()

    # Load and configure tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_path)
    
    # Define and add all special tokens requested by the user
    special_tokens_dict = {
        "additional_special_tokens": [
            "[CTX_CHURCH]", "[CTX_DAILY]", "[CTX_LEGAL]", 
            "[CTX_LIT]", "[CTX_EPIC]", "[CTX_SCIENCE]", "[GAP]",
            "·", ":"
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    splits_dir = Path(args.splits_dir)

    # Process and pack Train set
    print("Processing TRAIN...")
    train_lines = read_txt_lines(splits_dir / "train.txt", limit=args.limit)
    train_raw = encode_lines(train_lines, tokenizer)
    train_ds = Dataset.from_dict(train_raw).map(
        lambda x: group_texts(x, args.max_len),
        batched=True,
        desc=f"Packing train into blocks of {args.max_len}",
    )

    # Process and pack Validation set (test_a)
    print("Processing TEST_A...")
    test_a_lines = read_txt_lines(splits_dir / "test_a.txt", limit=args.limit)
    test_a_raw = encode_lines(test_a_lines, tokenizer)
    test_a_ds = Dataset.from_dict(test_a_raw).map(
        lambda x: group_texts(x, args.max_len),
        batched=True,
        desc=f"Packing test_a into blocks of {args.max_len}",
    )

    # Process fixed evaluation set (test_b)
    print("Processing TEST_B (fixed samples)...")
    test_b_records = read_jsonl(splits_dir / "test_b.jsonl", limit=args.limit)
    test_b_list = encode_test_b(test_b_records, tokenizer, args.max_len)
    test_b_ds = Dataset.from_list(test_b_list)

    # Save to disk
    ds = DatasetDict({"train": train_ds, "test_a": test_a_ds, "test_b": test_b_ds})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))

    print(f"Dataset preparation complete. Saved to: {out_dir}")


if __name__ == "__main__":
    main()
