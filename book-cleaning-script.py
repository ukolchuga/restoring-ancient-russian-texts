import os
import google.generativeai as genai
from dotenv import load_dotenv
import time
load_dotenv()
#put your api key in .env file
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

INPUT_FILE = "data/book.txt"
OUTPUT_FILE = "data/book_cleaned.txt"
LINES_PER_CHUNK = 500

def clean_text_with_gemini(text_chunk):
    prompt = f"Есть книга 17 века, только из-за ошибок OCR текст грязный, а мы хотим восстанавливать его с MLM, поэтому почисти его и выведи сюда, только сохраняй ту оригинальную орфографию, так как мы работаем с оригиналом. Выдай только исправленный текст:\n\n{text_chunk}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error while trying to connect to API: {e}")
        return None


def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i in range(0, len(lines), LINES_PER_CHUNK):
        chunk = "".join(lines[i:i + LINES_PER_CHUNK])
        print(f"Cleaning raws from {i} to {i + LINES_PER_CHUNK}...")

        cleaned_chunk = clean_text_with_gemini(chunk)

        if cleaned_chunk:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
                f_out.write(cleaned_chunk + "\n")
            print(f"Done, cleaned rows saved in {OUTPUT_FILE}")
#works, but too long, consider lowering waiting time, but be careful of TooManyRequests error
        time.sleep(10)


if __name__ == "__main__":
    main()