import re
import unicodedata

import pandas as pd

# ==========================================
# 📂 НАСТРОЙКИ
# ==========================================
INPUT_CSV = "gramoty_text_only.csv"
OUTPUT_TXT = "gramoty_final_cleaned.txt"


def gold_standard_spaced_clean(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # 0. Юникод-нормализация
    text = unicodedata.normalize("NFC", text)
    # 0.5 Убиваем кракозябры из Private Use Area (шрифтовые лигатуры)
    text = re.sub(r"[\uf000-\ufaff]", "", text)

    # 1. УДАЛЕНИЕ МЕТАТЕКСТА АРХЕОЛОГОВ
    garbage_patterns = [
        r"(?i)верхняя часть листа",
        r"(?i)нижняя часть листа",
        r"(?i)левая часть листа",
        r"(?i)правая часть листа",
        r"(?i)Приписка с поворотом на \d+ градусов\s*:",
        r"(?i)оборот\s*:",
        r"(?i)на обороте\s*:",
        r"(?i)Фрагмент[ы]?\s*\d+\s*и\s*\d+",
        r"(?i)Фрагмент[ы]?\s*\d+",
        r"(?i)мелкие изолированные отрезки не воспроизводятся",
        r"(?i)при повороте листа на \d+ читается еще одна запись.*?(?=зачеркнута\s*)зачеркнута",
        r"`\s*",  # Обратный апостроф от оцифровки
    ]
    for pattern in garbage_patterns:
        text = re.sub(pattern, "", text)

    # 2. Ошибки писца: удаляем текст в фигурных скобках {ло}
    text = re.sub(r"\{[^}]*\}", "", text)

    # 3. Реконструкция: убираем скобки [ ], ( ), ⟦ ⟧, оставляя сам текст
    text = re.sub(r"[\[\]\(\)⟦⟧]", "", text)

    # 4. Маркеры нечитаемых цифр (встречаются как "· - ·") -> превращаем в [GAP]
    text = re.sub(r"·\s*-\s*·", "[GAP]", text)

    # 5. Склеивание переносов строк (дефис + пробел)
    text = re.sub(r"[-‐‑]\s+", "", text)

    # 6. Лакуны: превращаем многоточия и длинные тире в [GAP]
    text = re.sub(r"(?:…|\.{2,}|[-‐‑–—−]{2,})", "[GAP]", text)

    # 7. Одиночные дефисы-обрывки "приклеиваем" как [GAP]
    text = re.sub(r"(?<=\s)[-‐‑](?=\w)", "[GAP]", text)
    text = re.sub(r"(?<=\w)[-‐‑](?=\s)", "[GAP]", text)
    text = re.sub(r"^[-‐‑](?=\w)", "[GAP]", text)
    text = re.sub(r"(?<=\w)[-‐‑]$", "[GAP]", text)

    # 8. Убиваем дефисы, которые прилипли к [GAP] (например: "м-[GAP]" -> "м[GAP]")
    text = re.sub(r"[-‐‑]+\[GAP\]", "[GAP]", text)
    text = re.sub(r"\[GAP\][-‐‑]+", "[GAP]", text)

    # 9. Выделители (точки, двоеточия и кресты в начале) отбиваем пробелами
    text = re.sub(r"\s*([·:+])\s*", r" \1 ", text)

    # 10. Тотальная зачистка поломанных скобок вокруг GAP
    text = re.sub(r"\[*GAP\]*", "[GAP]", text)

    # 11. Безопасное схлопывание дублей GAP (чтобы не съесть соседние пробелы)
    while "[GAP][GAP]" in text or "[GAP] [GAP]" in text:
        text = text.replace("[GAP][GAP]", "[GAP]").replace("[GAP] [GAP]", "[GAP]")

    # 12. Финальная чистка пробелов
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_valid_for_training(text):
    if not text or len(text) < 10:
        return False

    words = text.split()
    content_words = [w for w in words if w != "[GAP]" and w not in ["·", ":", "+"]]

    # 1. Минимум 3 реальных слова для контекста
    if len(content_words) < 3:
        return False

    # 2. Слишком "дырявый" текст отбрасываем (больше 60% масок)
    gap_count = text.count("[GAP]")
    if gap_count >= len(words) * 0.6:
        return False

    # 3. Удаляем азбуки (а б в г...)
    if re.search(r"\bа\s+б\s+в\s+г\b", text, re.IGNORECASE):
        return False

    # 4. Удаляем "склады" (слоговые упражнения)
    # Если двоеточий и точек слишком много по отношению к длине текста (> 15%)
    colons = text.count(":") + text.count("·")
    chars = len(text.replace(" ", ""))
    if chars > 0 and (colons / chars) > 0.15:
        return False

    return True


def main():
    print(f"🚀 Загрузка {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    target_col = "original_text_spaced"
    if target_col not in df.columns:
        print(f"❌ Ошибка: Колонка {target_col} не найдена!")
        return

    print("✨ Применение 'золотого стандарта' очистки для MLM...")
    df["clean_text"] = df[target_col].apply(gold_standard_spaced_clean)

    print("🧹 Фильтрация мусора (метатекст, азбуки, склады)...")
    mask = df["clean_text"].apply(is_valid_for_training)
    cleaned_data = df[mask]["clean_text"].unique().tolist()

    print(f"💾 Запись в {OUTPUT_TXT}...")
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_data))

    print(f"✅ Готово! Итоговое количество строк: {len(cleaned_data)}")


if __name__ == "__main__":
    main()
