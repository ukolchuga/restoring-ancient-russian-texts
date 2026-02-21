import requests
import re
import unicodedata
import csv


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
    csv_output_file = "bible_metadata.csv"

    total_lines = 0
    total_tokens = 0

    print(f"🚀 Downloading {len(books)} books:\n")
    TITLO_RANGE = range(0x0483, 0x0488)

    # Подготавливаем данные для CSV
    csv_data = [["Book_Filename", "Lines", "Tokens"]]

    with open(output_file, "w", encoding="utf-8") as f_out:
        for book in books:
            url = BASE_URL + book
            try:
                print(f"Downloading {book:<15}...", end=" ")
                response = requests.get(url)

                if response.status_code == 200:
                    text = response.text
                    clean_lines = []
                    file_tokens_count = 0

                    for line in text.split("\n"):
                        # Убираем номера строк и выделения
                        line = re.sub(r"^\d+\s*\|\s*", "", line)
                        line = re.sub(r"\*\*.*?\*\*", "", line)

                        if not line.strip():
                            continue

                        nfd_form = unicodedata.normalize("NFD", line)

                        clean_chars = []
                        for c in nfd_form:
                            # Убираем диакритику, но сохраняем титла
                            if unicodedata.category(c) != "Mn" or ord(c) in TITLO_RANGE:
                                clean_chars.append(c)

                        clean_text = unicodedata.normalize("NFC", "".join(clean_chars))
                        clean_text = clean_text.strip()

                        if len(clean_text) > 5:
                            clean_lines.append(clean_text)

                            # Подсчет токенов (слова + знаки препинания)
                            tokens_in_line = len(re.findall(r"\w+|[^\w\s]", clean_text))
                            file_tokens_count += tokens_in_line

                    file_lines_count = len(clean_lines)

                    # Записываем в общий файл с разделителем
                    f_out.write(f"\n\n--- {book} ---\n")
                    f_out.write("\n".join(clean_lines))

                    total_lines += file_lines_count
                    total_tokens += file_tokens_count

                    # Добавляем в массив для таблицы
                    csv_data.append([book, file_lines_count, file_tokens_count])

                    print(
                        f"✅ Ok ({file_lines_count} lines, {file_tokens_count} tokens)"
                    )
                else:
                    print(f"❌ Error {response.status_code}")
            except Exception as e:
                print(f"⚠️ Error: {e}")

    # Сохраняем статистику в CSV
    with open(csv_output_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)

    print("\n" + "=" * 60)
    print(f"🔥 ГОТОВО! Итоговый файл: {output_file}")
    print(f"📊 Статистика сохранена в: {csv_output_file}")
    print(f"Всего строк: {total_lines}")
    print(f"Всего токенов (TOROT style): {total_tokens}")
    print("=" * 60)

    print("\nExample of the cleaned text:")
    with open(output_file, "r", encoding="utf-8") as f:
        # Пропускаем первые 5 строк (там могут быть разделители и пустые строки)
        for _ in range(5):
            next(f)
        print(f.readline().strip())


if __name__ == "__main__":
    download_and_clean_ponomar_bible()
