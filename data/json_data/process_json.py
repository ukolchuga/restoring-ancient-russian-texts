import json
import os
import re

JSON_FILE = "DIACU_1.0.json"
OUTPUT_DIR = "diacu_extracted"


def sanitize_filename(name):
    name = re.sub(r"[^\w\-_]", "_", name)
    return re.sub(r"_{2,}", "_", name)[:50]


def main():
    if not os.path.exists(JSON_FILE):
        print(f"File {JSON_FILE} does not exist.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Folder was created: {OUTPUT_DIR}")

    print(f"Reading JSON...")

    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = data.get("Documents", [])
        print(f"Found documents: {len(docs)}")

        for idx, doc in enumerate(docs):
            title = doc.get("Title", "Untitled")
            if not title:
                title = doc.get("Original_title", f"doc_{idx}")

            content = doc.get("Content", "")

            raw_lines = content.splitlines()
            clean_lines = [line.strip() for line in raw_lines if line.strip()]

            safe_title = sanitize_filename(title)
            filename = f"{idx+1:03d}_{safe_title}.txt"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, "w", encoding="utf-8") as out:
                out.write("\n".join(clean_lines))

        print(f"\nAll files are in the folder: {OUTPUT_DIR}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
