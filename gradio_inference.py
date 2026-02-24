import gradio as gr
from transformers import pipeline

MODEL_ID = "AlexSychovUN/mini-bert-ancient-rus-v2"

print(f"⏳ Загрузка модели {MODEL_ID} из облака...")
fill_mask = pipeline(
    "fill-mask", model=MODEL_ID, tokenizer=MODEL_ID, device=-1, top_k=10
)
print("✅ Модель загружена!")

# 🔥 Добавили недостающий тег SCIENCE
TAGS = {
    "🏡 Бытовой (Грамоты)": "[CTX_DAILY]",
    "⚖️ Юридический (Акты)": "[CTX_LEGAL]",
    "⛪️ Церковный (Молитвы)": "[CTX_CHURCH]",
    "📚 Летописный (Повести)": "[CTX_LIT]",
    "⚔️ Эпический (Былины)": "[CTX_EPIC]",
    "🌿 Научный (Лечебники)": "[CTX_SCIENCE]",
}


def greedy_fill(text):
    current_text = text
    while "[MASK]" in current_text:
        results = fill_mask(current_text)
        preds = results[0] if isinstance(results[0], list) else results
        best_token = preds[0]["token_str"]
        current_text = current_text.replace("[MASK]", best_token, 1)
        current_text = current_text.replace(" ##", "").replace("##", "")
    return current_text


def restore_text_top_n(category_key, input_text, top_n):
    if "[MASK]" not in input_text:
        return "⚠️ Добавьте хотя бы один токен [MASK] в текст!", ""

    tag = TAGS[category_key]
    tagged_text = f"{tag} {input_text}"
    mask_count = tagged_text.count("[MASK]")

    results = fill_mask(tagged_text)
    first_mask_preds = results[0] if mask_count > 1 else results

    final_hypotheses = []
    log = f"📝 Исходник: {input_text}\\n"
    log += f"🔍 Развилка для первой маски (Топ-{top_n}):\\n"

    for i in range(min(top_n, len(first_mask_preds))):
        pred = first_mask_preds[i]
        token = pred["token_str"]
        score = pred["score"] * 100

        log += f"  Ветвь {i+1}: '{token}' (Уверенность: {score:.1f}%)\\n"

        partially_filled = tagged_text.replace("[MASK]", token, 1)
        fully_filled = greedy_fill(partially_filled)

        clean_text = fully_filled.replace(tag, "").strip()
        clean_text = clean_text.replace(" ##", "").replace("##", "")

        final_hypotheses.append(f"🏆 Вариант {i+1} [{score:.1f}%]:\\n{clean_text}")

    return "\\n\\n".join(final_hypotheses), log


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📜 ИИ-Реставратор Древнерусских Текстов")
    gr.Markdown(f"Модель **{MODEL_ID}** генерирует гипотезы восстановления лакун.")

    with gr.Row():
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=list(TAGS.keys()),
                value="🏡 Бытовой (Грамоты)",
                label="Стиль текста",
            )
            input_text = gr.Textbox(
                lines=4,
                placeholder="Вставьте текст с [MASK]...",
                label="Исходный текст",
            )
            top_n_slider = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                value=3,
                label="Количество вариантов (Топ-N)",
            )
            submit_btn = gr.Button("Сгенерировать гипотезы", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(lines=6, label="✨ Гипотезы реставрации")
            log_output = gr.Textbox(lines=5, label="🔍 Лог размышлений")

    # 🔥 Добавили пример для науки
    gr.Examples(
        examples=[
            ["🏡 Бытовой (Грамоты)", "И капѹстѹ все лѣто [MASK] и свеклѹ"],
            ["⚖️ Юридический (Акты)", "Аже смердъ ѹмреть то [MASK] кнѧзю"],
            ["⛪️ Церковный (Молитвы)", "То бо є истинное [MASK] [MASK] ."],
            ["📚 Летописный (Повести)", "Бѣ мѧтежь великъ и [MASK] по всеи тои странѣ"],
            ["⚔️ Эпический (Былины)", "Приехал [MASK] [MASK] к сыру дубу,"],
            [
                "🌿 Научный (Лечебники)",
                "А ѿ тоя болезни дай ему пити [MASK] , и тако исцелеет.",
            ],
        ],
        inputs=[category_dropdown, input_text],
        outputs=[output_text, log_output],
        fn=restore_text_top_n,
        cache_examples=False,
    )

    submit_btn.click(
        fn=restore_text_top_n,
        inputs=[category_dropdown, input_text, top_n_slider],
        outputs=[output_text, log_output],
    )

demo.launch()
