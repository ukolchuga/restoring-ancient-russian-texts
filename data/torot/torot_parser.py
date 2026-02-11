import os
import glob
from tqdm import tqdm


DATA_DIR = "./torot_data"
OUTPUT_FILE = "torot_corpus_final.txt"


def parse_conll_file(filepath):
    sentences = []
    current_sentence = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Empty line indicates end of sentence
            if not line:
                if current_sentence:
                    text = " ".join(current_sentence)
                    if len(text) > 2:
                        sentences.append(text)
                    current_sentence = []
                continue

            # Comments
            if line.startswith("#"):
                continue

            # Word - 2nd column
            parts = line.split("\t")
            if len(parts) > 1:
                word = parts[1]

                if word != "_" and word.strip():
                    current_sentence.append(word)

    if current_sentence:
        sentences.append(" ".join(current_sentence))
    return sentences


all_files = glob.glob(os.path.join(DATA_DIR, "*.conll"))
valid_sentences = []

print(f"Found files: {len(all_files)}")
for filepath in tqdm(all_files, desc="Parsing"):
    try:
        sentences = parse_conll_file(filepath)
        valid_sentences.extend(sentences)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

print(f"Total valid sentences extracted: {len(valid_sentences)}")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    for sent in valid_sentences:
        f_out.write(sent + "\n")

print(f"Wrote output to {OUTPUT_FILE}")
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    for i in range(3):
        print(f.readline().strip())
