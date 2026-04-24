import pandas as pd
import re
import os

INPUT_CSV = "data/epigraphica/epigraphica_full_data.csv"
OUTPUT_CSV = "data/epigraphica/epigraphica_clean_metadata.csv"
OUTPUT_TXT = "data/epigraphica/epigraphica_ready_for_bert.txt"

def parse_years(date_str):
    if not isinstance(date_str, str) or not date_str:
        return None, None, None
    years = re.findall(r'\d{3,4}', date_str)
    if not years:
        return None, None, None
    years = [int(y) for y in years]
    start_year = min(years)
    end_year = max(years)
    mean_year = sum(years) // len(years)
    century = (mean_year - 1) // 100 + 1
    return start_year, end_year, century

def clean_epigraphy_text(text):
    if not isinstance(text, str) or not text:
        return ""
    
    # 1. Удаляем технические маркеры (im, vac, text, текст) в любых комбинациях
    text = re.sub(r'(?i)(text|текст)\s*\d*[:：\s]*', ' ', text)
    text = re.sub(r'(?i)im\.|vac\.', ' ', text)
    
    # 2. Разделители и спецсимволы -> Пробел
    for char in ['|', '/', '¦', '*', '+', '~', '⁓', ':', '：']:
        text = text.replace(char, ' ')
    
    # 3. Лакуны -> [GAP]
    text = re.sub(r'\(…\)|…|\(-\)|-{2,}', ' [GAP] ', text)
    
    # 4. Удаляем зачеркнутое ⟦...⟧
    text = re.sub(r'⟦.*?⟧', '', text)
    
    # 5. Склейка слов (переносы)
    text = text.replace('⸗', '').replace('=', '')
    
    # 6. Раскрываем скобки [ ], ( ), { }
    text = re.sub(r'[\[\]\(\)\{\}]', '', text)
    
    # 7. Финальная чистка пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_quality_data(text):
    if not text: return False
    
    # Считаем буквы (кириллица + глаголица)
    letters = re.findall(r'[а-яА-ЯёЁ\u0400-\u052F\uA640-\uA69F\u2C00-\u2C5F]', text)
    if len(letters) < 15:
        return False
        
    # Убираем цифровой мусор (если цифр больше чем букв)
    digits = re.findall(r'\d', text)
    if len(digits) > len(letters):
        return False

    # Убираем греческий шум
    if re.search(r'[\u0370-\u03FF]', text):
        return False

    # Убираем "азбуки" (много одиночных букв)
    words = text.split()
    if len([w for w in words if len(w) == 1 and w not in 'аив']) > 3:
        return False
        
    # Убираем повторы (З З З)
    if re.search(r'(.)\s\1\s\1', text):
        return False

    return True

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found!")
        return

    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    print("Cleaning and Filtering...")
    df['clean_text'] = df['text'].apply(clean_epigraphy_text)
    
    # Парсим даты
    df[['start_year', 'end_year', 'century']] = df['date'].apply(
        lambda x: pd.Series(parse_years(x))
    )
    
    # Применяем фильтры
    df_valid = df[df['clean_text'].apply(is_quality_data)].copy()
    
    # Сохраняем результат
    df_valid.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        for _, row in df_valid.iterrows():
            f.write(f"{row['clean_text']}\n")
    
    print(f"Total entries before: {len(df)}")
    print(f"Total entries after: {len(df_valid)}")
    print(f"Cleanup finished. Data saved to {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
