import pandas as pd
import re

INPUT_CSV = "gramoty_train_fixed.csv"
OUTPUT_TXT = "gramoty_clean.txt"


def final_clean_gramoty(text):
    if not isinstance(text, str):
        return ""

 
    text = re.sub(
        r"(верхняя|нижняя|средняя)\s+часть\s+листа", "", text, flags=re.IGNORECASE
    )

 
    text = text.lstrip("\"' +`")

    
  
    text = text.replace("...", " [UNK] ")
    text = text.replace("…", " [UNK] ")


    text = text.replace("·-·", " [UNK] ")

 
    text = re.sub(r"\s+\[UNK\]\s+", " [UNK] ", text)

  
    text = re.sub(r",[IVXLCDM]+\d*,\w+,\d+.*$", "", text)

    return text.strip()


def main():
    print(f"📜 Обрабатываем грамоты из {INPUT_CSV}...")


    df = pd.read_csv(INPUT_CSV, dtype=str)


    text_col = next(
        (c for c in ["original_text_spaced", "text", "content"] if c in df.columns),
        None,
    )

    if not text_col:
        print("❌ Ошибка: Не нашел колонку с текстом!")
        return


    df["clean"] = df[text_col].apply(final_clean_gramoty)


    valid_rows = df[df["clean"].str.len() > 3]

    print(f"✅ Найдено {len(valid_rows)} строк. Сохраняем в {OUTPUT_TXT}...")


    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_rows["clean"].tolist()))

    print("🎉 Готово! Теперь грамоты чистые и с токенами [UNK].")


if __name__ == "__main__":
    main()
