import requests
import re
import unicodedata


def download_and_clean_ponomar_bible():
    BASE_URL = "https://raw.githubusercontent.com/typiconman/ponomar/master/Ponomar/languages/cu/bible/elis/"

    books = [
        "Gen.text",
        "Ex.text",
        "Lev.text",
        "Num.text",
        "Deut.text",
        "Josh.text",
        "Judg.text",
        "Ruth.text",
        "I_Kings.text",
        "II_Kings.text",
        "III_Kings.text",
        "IV_Kings.text",
        "I_Paral.text",
        "II_Paral.text",
        "I_Esdra.text",
        "II_Esdra.text",
        "Tobit.text",
        "Judith.text",
        "Esther.text",
        "Job.text",
        "Psalm.text",
        "Prov.text",
        "Eccles.text",
        "Song.text",
        "Wisd.text",
        "Sirach.text",
        "Isa.text",
        "Jerem.text",
        "Lamen.text",
        "Baruch.text",
        "Ezek.text",
        "Dan.text",
        "Hos.text",
        "Joel.text",
        "Amos.text",
        "Obad.text",
        "Jona.text",
        "Mica.text",
        "Nahum.text",
        "Habak.text",
        "Zeph.text",
        "Hagg.text",
        "Zech.text",
        "Mal.text",
        "I_Macc.text",
        "II_Macc.text",
        "III_Macc.text",
        "Mt.text",
        "Mk.text",
        "Lk.text",
        "Jn.text",
        "Acts.text",
        "Jas.text",
        "I_Pet.text",
        "II_Pet.text",
        "I_Jn.text",
        "II_Jn.text",
        "III_Jn.text",
        "Jude.text",
        "Rom.text",
        "I_Cor.text",
        "II_Cor.text",
        "Gal.text",
        "Eph.text",
        "Philip.text",
        "Col.text",
        "I_Thess.text",
        "II_Thess.text",
        "I_Tim.text",
        "II_Tim.text",
        "Tit.text",
        "Philemon.text",
        "Heb.text",
        "Apoc.text",
    ]

    output_file = "bible_full_clean.txt"
    total_lines = 0

    print(f"Downloading {len(books)} books:")
    TITLO_RANGE = range(0x0483, 0x0488)
    with open(output_file, "w", encoding="utf-8") as f_out:
        for book in books:
            url = BASE_URL + book
            try:
                print(f"Downloading {book}...", end=" ")
                response = requests.get(url)

                if response.status_code == 200:
                    text = response.text
                    clean_lines = []

                    for line in text.split("\n"):
                        line = re.sub(r"^\d+\s*\|\s*", "", line)
                        line = re.sub(r"\*\*.*?\*\*", "", line)

                        if not line.strip():
                            continue

                        nfd_form = unicodedata.normalize("NFD", line)

                        clean_chars = []
                        for c in nfd_form:

                            if unicodedata.category(c) != "Mn" or ord(c) in TITLO_RANGE:
                                clean_chars.append(c)

                        clean_text = unicodedata.normalize("NFC", "".join(clean_chars))
                        clean_text = clean_text.strip()

                        if len(clean_text) > 5:
                            clean_lines.append(clean_text)

                    f_out.write(f"\n\n--- {book} ---\n")
                    f_out.write("\n".join(clean_lines))
                    total_lines += len(clean_lines)
                    print(f"Ok ({len(clean_lines)} lines)")
                else:
                    print(f"Error {response.status_code}")
            except Exception as e:
                print(f"Error: {e}")

    print(f"\nFine! The file is saved: {output_file}")
    print(f"All lines: {total_lines}")
    print("Example of the cleaned text:")

    with open(output_file, "r", encoding="utf-8") as f:
        for _ in range(5):
            next(f)
        print(f.readline())


if __name__ == "__main__":
    download_and_clean_ponomar_bible()
