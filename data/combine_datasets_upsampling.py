import os
import pandas as pd
import re

OUTPUT_FILE = "final_dataset_ready.txt"

WEIGHTS = {
    "bible": 1,  # Bible = 45k lines
    "torot": 1,
    "literature": 5,
    "gramoty": 20,
    "legal": 40,
    "domostroy": 15,
}

SOURCES = [
    ("bible/bible_full_clean.txt", "bible", "[CTX_CHURCH]"),
    ("torot/torot_corpus_final.txt", "torot", "[CTX_BOOK]"),
    ("pushkinskij_texts/pushkinskij_full.txt", "literature", "[CTX_BOOK]"),
    ("birch_bark/gramoty_clean.txt", "gramoty", "[CTX_DAILY]"),
    ("sudebnic_1497_clean.txt", "legal", "[CTX_LEGAL]"),
    ("sudebnic_1550_clean.txt", "legal", "[CTX_LEGAL]"),
    ("sobornoe_izlozhenie_clean.txt", "legal", "[CTX_LEGAL]"),
    ("domostroy_clean.txt", "domostroy", "[CTX_BOOK]"),
]


def main():
    print("Writing combined dataset to", OUTPUT_FILE)
    total_lines_written = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for filename, source, tag in SOURCES:
            if not os.path.exists(filename):
                print("File not found:", filename)
                continue

            text = open(filename, "r", encoding="utf-8", errors="replace").read()
            lines = [line.strip() for line in text.splitlines() if line.strip()]

            multiplier = WEIGHTS.get(source, 1)
            print(
                f"{filename}: {len(text.splitlines())} lines, weight {multiplier}, tag: {tag}"
            )

            tagged_lines = [f"{tag} {line}" for line in lines]
            content_block = "\n".join(tagged_lines) + "\n"
            for _ in range(multiplier):
                f_out.write(content_block)
            total_lines_written += len(text.splitlines()) * multiplier

        print("Total lines written:", total_lines_written)


if __name__ == "__main__":
    main()
