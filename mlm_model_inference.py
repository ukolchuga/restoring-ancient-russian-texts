from transformers import pipeline

# Укажи путь к сохраненной модели
# Если ты только что обучил, это переменная OUTPUT_DIR
MODEL_PATH = "old_rus_bert_final_tags_upsampling"

print(f"⏳ Загружаю модель из {MODEL_PATH}...")

try:
    fill_mask = pipeline(
        "fill-mask",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=0,  # 0 для GPU, -1 для CPU
    )
except Exception as e:
    print(f"Ошибка загрузки: {e}")
    print(
        "Проверь, что папка существует и там есть файлы model.safetensors (или pytorch_model.bin) и config.json"
    )
    # Аварийный выход, если путь неверный
    exit()


def compare_styles_top3(sentence_template):
    """
    Прогоняет шаблон через 3 стиля и показывает Топ-3 для каждого.
    """
    styles = [
        ("[CTX_DAILY]", "📜 Грамота (Быт)"),
        ("[CTX_LEGAL]", "⚖️ Закон (Суд)"),
        ("[CTX_CHURCH]", "⛪ Церковь (Бог)"),
    ]

    print(f"\n" + "=" * 60)
    print(f"🔍 ТЕСТ ФРАЗЫ: '{sentence_template}'")
    print("=" * 60)

    for tag, style_name in styles:
        text = f"{tag} {sentence_template}"

        # top_k=3 дает 3 варианта
        results = fill_mask(text, top_k=3)

        print(f"\n{style_name}:")
        for i, res in enumerate(results, 1):
            token = res["token_str"]
            score = res["score"]
            print(f"   {i}. {token:<15} ({score:.1%})")


# --- СЦЕНАРИИ ТЕСТИРОВАНИЯ ---

# 1. Адресат (Кому кланяемся?)
# Грамоты: имена/родня. Закон: судьи/государи. Церковь: святые.
compare_styles_top3("поклоно ѿ [MASK] ко господину")

# 2. Денежно-Вещевой вопрос (Что взять?)
# Грамоты: конкретика (соль, рубль). Закон: штраф, пошлина.
compare_styles_top3("а [MASK] возми у него")

# 3. Власть и Сила (Во имя кого?)
# Церковь: Отца/Бога. Закон: Царя. Грамоты: (редко, мб Имени?)
compare_styles_top3("во имѧ [MASK]")

# 4. Действие (Что сделать с человеком?)
# Закон: судить/казнить/отпустить. Церковь: помиловать/благословить.
compare_styles_top3("и повелѣ [MASK] ихъ")

# 5. Проверка времени (Когда?)
# Церковь: во веки. Грамоты: днес/завтра.
compare_styles_top3("и бысть въ [MASK] день")
