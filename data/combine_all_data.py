import os
import random

from transformers import BertTokenizerFast, RobertaTokenizerFast

TOKENIZER_DIR = "../ancient_rus_tokenizer"
BPE_TOKENIZER_DIR = "../ancient_rus_tokenizer_BPE"


SOURCES_CONFIG = [
    # Daily
    {
        "type": "file",
        "path": "birch_bark/gramoty_clean.txt",
        "tag": "[CTX_DAILY]",
        "weight": 40,
    },
    {
        "type": "folder",
        "path": "json_data/diacu_DAILY",
        "tag": "[CTX_DAILY]",
        "weight": 15,
    },
    {"type": "folder", "path": "torot/torot_DAILY", "tag": "[CTX_DAILY]", "weight": 15},
    {
        "type": "folder",
        "path": "pushkinskij_texts/clean_texts/DAILY",
        "tag": "[CTX_DAILY]",
        "weight": 15,
    },
    {
        "type": "folder",
        "path": "NKRYA/NKRYA_TEXTS/DAILY",
        "tag": "[CTX_DAILY]",
        "weight": 15,
    },
    {
        "type": "file",
        "path": "epigraphica/epigraphica_ready_for_bert.txt",
        "tag": "[CTX_DAILY]",
        "weight": 20,
    },
    # Church
    {
        "type": "file",
        "path": "bible/bible_full_clean.txt",
        "tag": "[CTX_CHURCH]",
        "weight": 1,
    },
    {
        "type": "folder",
        "path": "json_data/diacu_CHURCH",
        "tag": "[CTX_CHURCH]",
        "weight": 1,
    },
    {
        "type": "folder",
        "path": "torot/torot_CHURCH",
        "tag": "[CTX_CHURCH]",
        "weight": 1,
    },
    {
        "type": "folder",
        "path": "pushkinskij_texts/clean_texts/CHURCH",
        "tag": "[CTX_CHURCH]",
        "weight": 1,
    },
    # Literature
    {
        "type": "folder",
        "path": "json_data/diacu_LIT",
        "tag": "[CTX_LIT]",
        "weight": 3,
    },
    {
        "type": "folder",
        "path": "torot/torot_LIT",
        "tag": "[CTX_LIT]",
        "weight": 5,
    },
    {
        "type": "folder",
        "path": "pushkinskij_texts/clean_texts/LIT",
        "tag": "[CTX_LIT]",
        "weight": 4,
    },
    {
        "type": "folder",
        "path": "NKRYA/NKRYA_TEXTS/LIT",
        "tag": "[CTX_LIT]",
        "weight": 4,
    },
    # Legal
    {
        "type": "folder",
        "path": "json_data/diacu_LEGAL",
        "tag": "[CTX_LEGAL]",
        "weight": 2,
    },
    {
        "type": "folder",
        "path": "torot/torot_LEGAL",
        "tag": "[CTX_LEGAL]",
        "weight": 20,
    },
    {
        "type": "folder",
        "path": "pushkinskij_texts/clean_texts/LEGAL",
        "tag": "[CTX_LEGAL]",
        "weight": 10,
    },
    {
        "type": "folder",
        "path": "NKRYA/NKRYA_TEXTS/LEGAL",
        "tag": "[CTX_LEGAL]",
        "weight": 10,
    },
    # Epic
    {
        "type": "folder",
        "path": "bilini",
        "tag": "[CTX_EPIC]",
        "weight": 10,
    },
    {
        "type": "folder",
        "path": "pushkinskij_texts/clean_texts/EPIC",
        "tag": "[CTX_EPIC]",
        "weight": 10,
    },
    # Science
    {
        "type": "folder",
        "path": "pushkinskij_texts/clean_texts/SCIENCE",
        "tag": "[CTX_SCIENCE]",
        "weight": 50,
    },
    {
        "type": "folder",
        "path": "NKRYA/NKRYA_TEXTS/SCIENCE",
        "tag": "[CTX_SCIENCE]",
        "weight": 50,
    },
    {
        "type": "file",
        "path": "ustav/ustav_for_training_2.txt",
        "tag": "[CTX_SCIENCE]",
        "weight": 15,
    },
]

OUTPUT_FILE = "final_ancient_rus_dataset.txt"


def load_and_process_source(config, tokenizer, bpe_tokenizer):
    path = config["path"]
    src_type = config["type"]
    tag = config["tag"]
    weight = config["weight"]

    if not os.path.exists(path):
        print(f"⚠️ Warning: {path} was not found! Skip.")
        return [], 0, 0

    unique_lines = set()
    if src_type == "file":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    unique_lines.add(clean_line)
    elif src_type == "folder":
        if not os.path.isdir(path):
             print(f"⚠️ Warning: {path} is not a directory! Skip.")
             return [], 0, 0
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt")]
        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    clean_line = line.strip()
                    if clean_line:
                        unique_lines.add(clean_line)

    tagged_lines = []
    source_tokens = 0
    source_tokens_bpe = 0

    print(f"Processing {path}...")
    for line in unique_lines:
        full_text = f"{tag} {line}"
        
        # Считаем токены только если токенизаторы загружены
        if tokenizer:
            tokens_count = len(tokenizer.encode(full_text, add_special_tokens=False))
            source_tokens += tokens_count * weight
            
        if bpe_tokenizer:
            tokens_count_bpe = len(bpe_tokenizer.encode(full_text, add_special_tokens=False))
            source_tokens_bpe += tokens_count_bpe * weight
            
        tagged_lines.extend([full_text] * weight)

    return tagged_lines, source_tokens, source_tokens_bpe


def main():
    print("Tokenizers loading...")
    tokenizer = None
    bpe_tokenizer = None
    
    try:
        if os.path.exists(TOKENIZER_DIR):
            tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)
            print("✅ BERT Tokenizer loaded.")
    except Exception as e:
        print(f"⚠️ BERT Tokenizer could not be loaded: {e}")

    try:
        if os.path.exists(BPE_TOKENIZER_DIR):
            bpe_tokenizer = RobertaTokenizerFast.from_pretrained(BPE_TOKENIZER_DIR)
            print("✅ BPE Tokenizer loaded.")
    except Exception as e:
        print(f"⚠️ BPE Tokenizer could not be loaded: {e}")

    if not tokenizer and not bpe_tokenizer:
        print("💡 Bootstrap mode: creating dataset without token statistics.")

    print("Combining the data...")
    all_final_lines = []
    stats = {}
    stats_bpe = {}

    for source_cfg in SOURCES_CONFIG:
        tag = source_cfg["tag"]
        lines, token_count, token_count_bpe = load_and_process_source(
            source_cfg, tokenizer, bpe_tokenizer
        )
        all_final_lines.extend(lines)

        if tokenizer:
            stats[tag] = stats.get(tag, 0) + token_count
        if bpe_tokenizer:
            stats_bpe[tag] = stats_bpe.get(tag, 0) + token_count_bpe

    if stats or stats_bpe:
        print("\n" + "=" * 50)
        if stats:
            print("BERT tokens statistics: ")
            total_tokens = sum(stats.values())
            for tag, count in stats.items():
                percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
                print(f"{tag:<15}: {count:>12,} tokens ({percentage:>5.2f}%)")
            print(f"Total BERT tokens: {total_tokens:,}")
        
        if stats_bpe:
            print("\nBPE tokens statistics:")
            total_tokens_bpe = sum(stats_bpe.values())
            for tag, count in stats_bpe.items():
                percentage = (count / total_tokens_bpe) * 100 if total_tokens_bpe > 0 else 0
                print(f"{tag:<15}: {count:>12,} BPE tokens ({percentage:>5.2f}%)")
            print(f"Total BPE tokens: {total_tokens_bpe:,}")
        print("=" * 50)

    print(f"Shuffling {len(all_final_lines):,} lines...")
    random.seed(42)
    random.shuffle(all_final_lines)

    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(all_final_lines))

    print("\nAncient Rus Corpus is ready!! ✨")


if __name__ == "__main__":
    main()
