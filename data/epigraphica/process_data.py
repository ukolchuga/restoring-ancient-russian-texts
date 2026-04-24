import pandas as pd
import re
import os
import unicodedata

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
    
    # 0. NFC нормализация (Критично для титл и BPE)
    text = unicodedata.normalize('NFC', text)

    # 1. Предварительная маркировка лакун (кириллицей, чтобы не съела очистка латиницы)
    # vac. (пустое место) и im. (рисунок) — это по сути исторические лакуны
    text = re.sub(r'(?i)\|?im\.|\|?vac\.', ' МАРКЕРГАП ', text)
    # Ловим (...), [---], ..., и множественные тире
    text = re.sub(r'(\.[\s·]*){3,}|…|\([\.…\-]+\)|\[[\.…\-]+\]|([\-‐‑–—−][\s·]*){2,}|·-·', ' МАРКЕРГАП ', text)

    # 2. Убираем технические маркеры сайта Epigraphica
    text = re.sub(r'(?i)(text|текст)\s*\d*[:：\s]*', ' ', text)
    
    # 3. Разделяем слипшиеся слова ( Scriptio Continua )
    # Окружаем древние разделители и переносы строк (|) пробелами
    text = re.sub(r'([·⁘⋮✝+])', r' \1 ', text)
    text = text.replace('|', ' ').replace('/', ' ').replace('¦', ' ')

    # 4. Раскрываем скобки восстановления и удаления
    text = re.sub(r'⟦.*?⟧', '', text) # Удаляем зачеркнутое
    text = re.sub(r'[\[\]\(\)\{\}<>]', '', text)
    
    # Склейка разорванных слов (переносы)
    text = text.replace('⸗', '').replace('=', '')

    # 5. Очистка от мусора
    text = re.sub(r'[a-zA-Z]', '', text) # Удаляем латиницу (маркер МАРКЕРГАП выживет)
    text = re.sub(r'\d', '', text)       # Современные цифры
    # Греческий алфавит вырезаем, если он встречается фрагментарно
    text = re.sub(r'[\u0370-\u03FF]', '', text) 

    # 6. ФИНАЛЬНЫЙ ЭТАП: ВОЗВРАЩАЕМ [GAP]
    text = text.replace("МАРКЕРГАП", " [GAP] ")
    
    # Схлопываем идущие подряд [GAP] и лишние пробелы
    text = re.sub(r'(\s*\[GAP\]\s*)+', ' [GAP] ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def is_quality_data(text):
    if not text: return False
    
    # Считаем буквы (кириллица + глаголица)
    letters = re.findall(r'[а-яА-ЯёЁ\u0400-\u052F\uA640-\uA69F\u2C00-\u2C5F]', text)
    if len(letters) < 12: # Слегка снизили порог для коротких граффити
        return False
        
    # Убираем "азбуки" (много одиночных букв)
    words = text.split()
    if len([w for w in words if len(w) == 1 and w not in 'аив·:']) > 4:
        return False
        
    # Убираем повторы (З З З)
    if re.search(r'(.)\s\1\s\1', text):
        return False

    # Проверка на "дырявость"
    gap_count = text.count("[GAP]")
    if gap_count > 0 and gap_count >= len(words) * 0.7:
        return False

    return True

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found!")
        return

    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    print("🔧 Cleaning epigraphica texts for BPE...")
    df['clean_text'] = df['text'].apply(clean_epigraphy_text)
    
    # Парсим даты
    print("⏳ Processing dates...")
    df[['start_year', 'end_year', 'century']] = df['date'].apply(
        lambda x: pd.Series(parse_years(x))
    )
    
    # Применяем фильтры
    print("🎯 Filtering low quality entries...")
    df_valid = df[df['clean_text'].apply(is_quality_data)].copy()
    
    # Сохраняем результат
    df_valid.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        for _, row in df_valid.iterrows():
            f.write(f"{row['clean_text']}\n")
    
    print("\n" + "=" * 50)
    print(f"✅ Total entries before: {len(df)}")
    print(f"🚀 Total entries after:  {len(df_valid)}")
    print(f"💾 Clean text saved to: {OUTPUT_TXT}")
    print("=" * 50)

if __name__ == "__main__":
    main()
