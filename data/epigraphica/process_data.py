import os
import re
import unicodedata

import pandas as pd

# ==========================================
# 📂 НАСТРОЙКИ
# ==========================================
INPUT_CSV = "data/epigraphica/epigraphica_full_data.csv"
OUTPUT_TXT = "gramoty_final_cleaned.txt"


def clean_epigraphy_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # 0. Юникод-нормализация
    text = unicodedata.normalize("NFC", text)

    # 1. Специфический мусор эпиграфики
    # Убиваем вставки вроде "text 1:", "text 2:" и т.д.
    text = re.sub(r"(?i)text\s*\d+\s*:", "", text)

    # Склеиваем слова, разбитые переносом строки (символы ⸗, =, ~, ̴)
    # Например: "Ми⸗ халу" -> "Михалу"
    text = re.sub(r"[⸗=~̴]\s*", "", text)

    # 2. Лакуны: превращаем многоточия, в т.ч. в скобках (...), и длинные тире в [GAP]
    text = re.sub(r"\([\.…-]+\)", "[GAP]", text)
    text = re.sub(r"\[[\.…-]+\]", "[GAP]", text)
    text = re.sub(r"(?:…|\.{2,}|[-‐‑–—−]{2,})", "[GAP]", text)

    # 3. Реконструкция: убираем ВСЕ виды скобок, оставляя сам восстановленный текст
    text = re.sub(r"[\[\]\(\)\{\}\<\>⟦⟧⟨⟩]", "", text)

    # 4. Удаляем пунктуацию и разделители (как договорились для MLM)
    text = re.sub(r"[·:×|¦⁞]", " ", text)

    # 5. Одиночные дефисы-обрывки "приклеиваем" как [GAP]
    text = re.sub(r"(?<=\s)[-‐‑](?=\w)", "[GAP]", text)
    text = re.sub(r"(?<=\w)[-‐‑](?=\s)", "[GAP]", text)
    text = re.sub(r"^[-‐‑](?=\w)", "[GAP]", text)
    text = re.sub(r"(?<=\w)[-‐‑]$", "[GAP]", text)

    # 6. Убиваем дефисы, которые прилипли к [GAP]
    text = re.sub(r"[-‐‑]+\[GAP\]", "[GAP]", text)
    text = re.sub(r"\[GAP\][-‐‑]+", "[GAP]", text)

    # 7. Тотальная зачистка поломанных скобок вокруг GAP
    text = re.sub(r"\[*GAP\]*", "[GAP]", text)

    # 8. Безопасное схлопывание дублей GAP
    while "[GAP][GAP]" in text or "[GAP] [GAP]" in text:
        text = text.replace("[GAP][GAP]", "[GAP]").replace("[GAP] [GAP]", "[GAP]")

    # 9. Финальная чистка лишних пробелов
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_valid_for_training(text):
    if not text or len(text) < 10:
        return False

    words = text.split()
    content_words = [w for w in words if w != "[GAP]"]

    # Минимум 3 реальных слова для контекста
    if len(content_words) < 3:
        return False

    # Слишком "дырявый" текст отбрасываем (больше 60% масок)
    gap_count = text.count("[GAP]")
    if gap_count >= len(words) * 0.6:
        return False

    return True


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Файл {INPUT_CSV} не найден!")
        return

    print("🚀 Загрузка данных эпиграфики...")
    df = pd.read_csv(INPUT_CSV)

    cleaned_inscriptions = []

    print("✨ Очистка и фильтрация...")
    for index, row in df.iterrows():
        # Берем оригинальный текст. Если его нет — берем реконструкцию
        raw_text = row.get("text", "")
        if pd.isna(raw_text) or str(raw_text).strip() == "":
            raw_text = row.get("reconstruction", "")

        clean_text = clean_epigraphy_text(str(raw_text))

        if is_valid_for_training(clean_text):
            # Добавляем наш спец-токен контекста церкви/эпиграфики, если хочешь
            # (Можно закомментировать строку ниже, если не хочешь использовать тег)
            clean_text = f"[CTX_CHURCH] {clean_text}"
            cleaned_inscriptions.append(clean_text)

    # Добавляем в конец нашего основного файла
    print(f"💾 Добавление {len(cleaned_inscriptions)} новых строк в {OUTPUT_TXT}...")
    with open(OUTPUT_TXT, "a", encoding="utf-8") as f:
        f.write("\n")  # Отбиваем пустой строкой на всякий случай
        f.write("\n".join(cleaned_inscriptions))

    print("✅ Готово! Эпиграфика успешно интегрирована в датасет.")


if __name__ == "__main__":
    main()
