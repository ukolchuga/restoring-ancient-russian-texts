import pandas as pd
import numpy as np
import re
import os
import unicodedata

# ==========================================
# 📂 НАСТРОЙКИ ПУТЕЙ
# ==========================================
INPUT_CSV = "gramoty_text_only.csv"
OUTPUT_TXT = "gramoty_clean.txt"
OUTPUT_CSV = "gramoty_metadata.csv"


def unified_clean_pipeline(text):
    if pd.isna(text):
        return ""
    text = str(text)

    # 0. ЮНИКОД-НОРМАЛИЗАЦИЯ
    text = unicodedata.normalize('NFC', text)

    # 1. ПРЕДВАРИТЕЛЬНАЯ МАРКИРОВКА ЛАКУН
    # Ловим все: ..., ---, . . ., - - -, [---], (....)
    # Используем кириллический маркер, чтобы его не съела очистка латиницы
    text = re.sub(r"(\.[\s·]*){3,}|([\-‐‑–—−][\s·]*){2,}|…|·-·", " МАРКЕРГАП ", text)

    # 2. УБИРАЕМ МУСОР (lookahead для защиты маркера)
    text = re.sub(r"\{[^}]+\}", "", text)
    
    intro_garbage = [
        r"(Кроме того|Далее|Приписка|Подпись|Текст|Запись|Фрагмент).*?(?=МАРКЕРГАП|:)\s*",
        r"(Начальная|Конечная|Средняя|Верхняя|Нижняя|Левая|Правая)\s+(часть|сторона|полоса).*?(?=МАРКЕРГАП|:)\s*",
        r"(В|Во)\s+(первом|втором|третьем|четвертом|пятом)\s+столбце.*?(?=МАРКЕРГАП|:)\s*",
        r"Запись читается.*?:",
        r"На\s+(нижней|верхней|другой)\s+(кромке|поле|полосе|стороне|обороте).*?(?=МАРКЕРГАП|:)\s*",
        r"При\s+(повороте|перевороте).*?(?=МАРКЕРГАП|:)\s*",
        r".*?листа кверху ногами:\s*",
    ]
    for pattern in intro_garbage:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    garbage_patterns = [
        r"(верхняя|нижняя|средняя|левая|правая|начальная|конечная)\s+(часть|колонка)(\s+листа)?",
        r"в\s+(левой|правой)\s+части",
        r"(Первый|Второй|Третий|Четвертый|Пятый|Шестой)(\s+(дворъ|оп|фрагмент|столбец))?",
        r"Оборот(\s+другой почерк)?",
        r"(Основная|Дополнительная)\s+часть",
        r"Строка(\s*:\s*[а-яА-ЯёЁa-zA-Z])?",
        r"После разрыва",
        r"вероятно,[^а-яА-Яѣѧѿꙑїѥюꙩѡ]*[а-яА-ЯёЁ\s]+",
        r"автор сперва собирался.*?передумал",
        r"на другом\s*(фрагменте|обрывке)?\s*(?=МАРКЕРГАП)?",
        r"от остальных записей этого.*?(?=МАРКЕРГАП|\s)",
        r"для следующей строки.*?:",
        r"(фрагмент|сторона|столбец|фрагменты)\s+[А-Яа-яA-Za-z0-9\sи,]+",
        r"соответствие строк лишь предположительно",
        r"почерк другой",
        r"на обороте",
        r"вверху рисунка",
        r"слева от рисунка",
        r"мелкие изолированные отрезки не воспроизводятся",
        r"перевод неясен",
        r"текст утрачен",
        r"№\s*\d+[а-я]?",
        r"Стр\.\s*\d+",
    ]
    for pattern in garbage_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 3. ОЧИСТКА СИМВОЛОВ
    text = re.sub(r",[IVXLCDM]+\d*,\w+,\d+.*$", "", text)
    
    # Раскрываем скобки (МАРКЕРГАП их не боится)
    text = re.sub(r"[\[\]\(\)⟦⟧]", "", text)
    
    text = re.sub(r"[\uf000-\ufaff]", "", text)
    text = re.sub(r"[|¦°×+?]", " ", text)
    
    # Удаляем латиницу (МАРКЕРГАП кириллический, он выживет)
    text = re.sub(r"[a-zA-Z]", "", text)
    text = re.sub(r"\d", "", text)

    # 4. СКЛЕИВАЕМ СЛОВА И ИЗОЛИРУЕМ ПУНКТУАЦИЮ
    pattern = r"(\w)\s*[-‐‑–—−]\s*(\w)"
    for _ in range(3):
        text = re.sub(pattern, r"\1\2", text)
    
    text = re.sub(r"([·:])", r" \1 ", text)

    # 5. ФИНАЛЬНЫЙ ЭТАП: ВОЗВРАЩАЕМ [GAP]
    text = text.replace("МАРКЕРГАП", " [GAP] ")
    
    # Схлопываем идущие подряд [GAP] и лишние пробелы
    text = re.sub(r"(\s*\[GAP\]\s*)+", " [GAP] ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    print(f"📂 Читаем данные из: {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV, dtype=str)
    except Exception as e:
        print(f"❌ Ошибка чтения CSV: {e}")
        return

    print("🔧 Чистим тексты (v6: Bulletproof GAP recovery)...")
    text_col = next((c for c in ["original_text_spaced", "original_text_raw", "content", "text"] if c in df.columns), None)
    if not text_col:
        print("❌ Ошибка: Не нашел колонку с текстом!")
        return

    df["clean"] = df[text_col].apply(unified_clean_pipeline)

    def is_valid_line(text):
        if not text or len(text) < 15: 
            return False
        
        letters = re.findall(r"[а-яА-ЯёЁ\u0400-\u052F\uA640-\uA69F]", text)
        if len(letters) < 8: 
            return False
            
        words = text.split()
        gap_count = text.count("[GAP]")
        
        # Допускаем до 80% лакун, если букв достаточно
        if gap_count > 0 and gap_count >= len(words) * 0.8:
            return False
            
        return True

    valid_rows = df[df["clean"].apply(is_valid_line)].copy()

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_rows["clean"].tolist()))

    print(f"💾 Сохранено {len(valid_rows)} строк.")
    valid_rows.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("🎉 ГОТОВО! Теперь [GAP] точно должны быть в файле.")
    print("=" * 60)


if __name__ == "__main__":
    main()
