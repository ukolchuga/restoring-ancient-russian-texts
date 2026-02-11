import os

def merge_manual_files_no_filter():
    input_folder = "raw_texts"
    output_file = "pushkinskij_full.txt"


    all_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

    total_lines = 0

    print(f"Found files: {len(all_files)}")

    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in all_files:
            filepath = os.path.join(input_folder, filename)

            with open(filepath, "r", encoding="utf-8") as infile:
                lines = infile.readlines()

            outfile.write(f"\n\n--- {filename} ---\n\n")

            file_lines = 0
            for line in lines:
                clean_line = line.strip()
                if clean_line:
                    outfile.write(clean_line + "\n")
                    file_lines += 1

            print(f"File {filename}: added {file_lines} lines.")
            total_lines += file_lines

    print(f"\nFine! File: {output_file}")
    print(f"All lines: {total_lines}")


if __name__ == "__main__":
    merge_manual_files_no_filter()