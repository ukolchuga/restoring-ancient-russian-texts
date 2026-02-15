import re
import os
from tqdm import tqdm 


INPUT_FILE = "final_dataset_ready.txt"
OUTPUT_FILE = "final_dataset_clean.txt"


def advanced_clean_text(text):
    if not isinstance(text, str): return ""

  
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"[\ue000-\uf8ff]", "", text)

    
    replacements = {
        "A": "А", "a": "а", "B": "В", "E": "Е", "e": "е",
        "K": "К", "k": "к", "M": "М", "H": "Н", "O": "О", "o": "о",
        "P": "Р", "p": "р", "C": "С", "c": "с", "T": "Т",
        "y": "у", "X": "Х", "x": "х"
    }
    for lat, cyr in replacements.items():
        text = text.replace(lat, cyr)


    text = re.sub(r"[\u0300-\u036f]", "", text)

   
    TITLO = r"[\u0483-\u0489]"
    abbrev_map = {
        rf"\bбг\b(?!{TITLO})": "богъ",
        rf"\bгд\b(?!{TITLO})": "господь",
        rf"\bсн\b(?!{TITLO})": "сынъ",
        rf"\bхс\b(?!{TITLO})": "христосъ",
        rf"\bгн\b(?!{TITLO})": "господинъ",
    }
    for pattern, repl in abbrev_map.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    
    text = re.sub(r"\s+", " ", text).strip()

   

    # [СТХ_СНURСН] -> [CTX_CHURCH]
    text = text.replace('[СТХ_СНURСН]', '[CTX_CHURCH]')

    # [СТХ_LЕGАL] -> [CTX_LEGAL] (L, G остались латиницей, E, A стали кириллицей)
    text = text.replace('[СТХ_LЕGАL]', '[CTX_LEGAL]')

    # [СТХ_DАILY] -> [CTX_DAILY] (D, I, L, Y латиница, A кириллица)
    text = text.replace('[СТХ_DАILY]', '[CTX_DAILY]')

    # [СТХ_ВООК] -> [CTX_BOOK] (B, O, O, K стали кириллицей)
    text = text.replace('[СТХ_ВООК]', '[CTX_BOOK]')

    # [UNК] -> [UNK] (K стала кириллицей)
    text = text.replace('[UNК]', '[UNK]')

    return text


if not os.path.exists(INPUT_FILE):
    print(
        f"Error: File {INPUT_FILE} is not found! First create it."
    )
else:
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, open(
        OUTPUT_FILE, "w", encoding="utf-8"
    ) as f_out:

        lines = f_in.readlines()
        cleaned_count = 0

        for line in tqdm(lines, desc="Lines processing"):
            if line.startswith("---"):
                f_out.write(line)
                continue

            original = line
            cleaned = advanced_clean_text(line)

            if len(cleaned) > 1:
                f_out.write(cleaned + "\n")
                cleaned_count += 1

    print(f"\nClean file: {OUTPUT_FILE}")
    print(f"Saved lines: {cleaned_count}")
   
