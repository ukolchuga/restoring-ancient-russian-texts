import gradio as gr
from transformers import pipeline


MODELS = {
    "🛡️ Базовый Mini-BERT": "AlexSychovUN/mini-bert-ancient-rus-v2",
    "🏹 Mini-BERT (Span Masking)": "AlexSychovUN/mini-bert-ancient-rus-span-collator-v1",
    "👑 Mini-RoBERTa (BPE)": "AlexSychovUN/mini-roberta-ancient-rus",
}

print("⏳ Загрузка моделей в память (это займет немного времени)...")
pipes = {}
for name, model_id in MODELS.items():
    pipes[name] = pipeline(
        "fill-mask", model=model_id, tokenizer=model_id, device=-1, top_k=10
    )
print("✅ Все модели успешно загружены!")

TAGS = {
    "🏡 Бытовой (Грамоты)": "[CTX_DAILY]",
    "⚖️ Юридический (Акты)": "[CTX_LEGAL]",
    "⛪️ Церковный (Молитвы)": "[CTX_CHURCH]",
    "📚 Летописный (Повести)": "[CTX_LIT]",
    "⚔️ Эпический (Былины)": "[CTX_EPIC]",
    "🌿 Научный (Лечебники)": "[CTX_SCIENCE]",
}


def clean_tokens(text, is_roberta):

    if is_roberta:
        return text.replace(" Ġ", " ").replace("Ġ", "")
    else:
        return text.replace(" ##", "").replace("##", "")


def restore_text_top_n(model_choice, category_key, input_text, top_n):
    if "[MASK]" not in input_text:
        return "⚠️ Добавьте хотя бы один токен [MASK] в текст!", ""

    tag = TAGS[category_key]
    is_roberta = "RoBERTa" in model_choice
    mask_token = "<mask>" if is_roberta else "[MASK]"

    working_text = input_text.replace("[MASK]", mask_token)
    tagged_text = f"{tag} {working_text}"
    mask_count = tagged_text.count(mask_token)

    current_pipe = pipes[model_choice]
    results = current_pipe(tagged_text)

    first_mask_preds = results[0] if mask_count > 1 else results

    final_hypotheses = []
    log = f"🧠 Модель: {model_choice}\n"
    log += f"📝 Исходник: {input_text}\n"
    log += f"🔍 Развилка для первой маски (Топ-{top_n}):\n"

    for i in range(min(top_n, len(first_mask_preds))):
        pred = first_mask_preds[i]
        token = clean_tokens(pred["token_str"], is_roberta)
        score = pred["score"] * 100

        log += f"  Ветвь {i+1}: '{token}' (Уверенность: {score:.1f}%)\n"

        current_filled_text = tagged_text.replace(mask_token, pred["token_str"], 1)

        while mask_token in current_filled_text:
            temp_res = current_pipe(current_filled_text)
            next_preds = temp_res[0] if isinstance(temp_res[0], list) else temp_res
            best_token = next_preds[0]["token_str"]
            current_filled_text = current_filled_text.replace(mask_token, best_token, 1)

        clean_text = current_filled_text.replace(tag, "").strip()
        clean_text = clean_tokens(clean_text, is_roberta)

        final_hypotheses.append(f"🏆 Вариант {i+1} [{score:.1f}%]:\n{clean_text}")

    return "\n\n".join(final_hypotheses), log


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📜 ИИ-Реставратор Древнерусских Текстов")
    gr.Markdown(
        "Выберите архитектуру нейросети, стиль документа и вставьте текст с пропусками `[MASK]`, чтобы получить гипотезы восстановления."
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="👑 Mini-RoBERTa (BPE)",
                label="Архитектура нейросети",
            )
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
            log_output = gr.Textbox(lines=5, label="🔍 Лог размышлений (Logits)")

    gr.Examples(
        examples=[
            [
                "👑 Mini-RoBERTa (BPE)",
                "🏡 Бытовой (Грамоты)",
                "Поклонъ ѿ петра ко [MASK] . а серебро ми отдай.",
            ],
            [
                "🏹 Mini-BERT (Span Masking)",
                "⚖️ Юридический (Акты)",
                "Аже кто оубиеть [MASK] , то платити виру.",
            ],
            [
                "👑 Mini-RoBERTa (BPE)",
                "⛪️ Церковный (Молитвы)",
                "И рече господь къ [MASK] своимъ.",
            ],
            [
                "🛡️ Базовый Mini-BERT",
                "📚 Летописный (Повести)",
                "И пошелъ князь игорь на [MASK] землю со своею дружиною.",
            ],
            [
                "👑 Mini-RoBERTa (BPE)",
                "⚔️ Эпический (Былины)",
                "Выезжал добрый молодец из города на добромъ [MASK] .",
            ],
            [
                "🏹 Mini-BERT (Span Masking)",
                "🌿 Научный (Лечебники)",
                "Аще кто боленъ главою, дай пити ему [MASK] от травы.",
            ],
            [
                "👑 Mini-RoBERTa (BPE)",
                "🏡 Бытовой (Грамоты)",
                "И капѹстѹ все лѣто [MASK] и свеклѹ",
            ],
        ],
        inputs=[model_dropdown, category_dropdown, input_text],
        outputs=[output_text, log_output],
        fn=restore_text_top_n,
        cache_examples=False,
    )

    submit_btn.click(
        fn=restore_text_top_n,
        inputs=[model_dropdown, category_dropdown, input_text, top_n_slider],
        outputs=[output_text, log_output],
    )

if __name__ == "__main__":
    demo.launch()
