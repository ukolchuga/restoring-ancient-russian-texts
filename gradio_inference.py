import gradio as gr
from transformers import pipeline


MODEL_PATH = "old_rus_bert_final_tags_upsampling"

print(f"Loading model {MODEL_PATH}...")

try:
    fill_mask = pipeline(
        "fill-mask",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=0,  # 0 = GPU, -1 = CPU
    )
except Exception as e:
    print(f"Error: {e}")
    print("Check the model path!")
    fill_mask = None


def predict_old_russian(text, style_name):
    if fill_mask is None:
        return "Model is not loaded", "Error"

    style_map = {
        "📜 Грамота (Быт/Береста)": "[CTX_DAILY]",
        "⚖️ Закон (Суд/Право)": "[CTX_LEGAL]",
        "⛪ Церковь (Библия)": "[CTX_CHURCH]",
        "📖 Книжность (Летопись)": "[CTX_BOOK]",
    }

    tag = style_map.get(style_name, "[CTX_DAILY]")

    if "[MASK]" not in text:
        return (
            {"Error": 1.0},
            "Пожалуйста, добавьте токен [MASK] в текст, чтобы модель знала, что угадывать.",
        )

    full_input = f"{tag} {text}"

    results = fill_mask(full_input, top_k=5)

    output_dict = {}
    for res in results:
        word = res["token_str"]
        score = res["score"]
        output_dict[word] = score

    top_word = results[0]["token_str"]
    explanation = f"Контекст: {tag}\nЛучший вариант: {top_word}"

    return output_dict, explanation


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🦢 Old Russian BERT: Демонстрация
        **Восстановление древнерусских текстов с учетом контекста.** Выберите стиль, напишите фразу с пропуском `[MASK]` и посмотрите, как нейросеть понимает историю.
        """
    )

    with gr.Row():
        with gr.Column():

            input_text = gr.Textbox(
                label="Введите фразу с [MASK]",
                placeholder="поклоно ѿ [MASK] ко господину",
                lines=2,
            )

            style_radio = gr.Radio(
                choices=[
                    "📜 Грамота (Быт/Береста)",
                    "⚖️ Закон (Суд/Право)",
                    "⛪ Церковь (Библия)",
                    "📖 Книжность (Летопись)",
                ],
                value="📜 Грамота (Быт/Береста)",
                label="Выберите контекст (Стиль)",
            )

            btn = gr.Button("🔮 Предсказать", variant="primary")

            gr.Examples(
                examples=[
                    ["поклоно ѿ [MASK] ко господину", "📜 Грамота (Быт/Береста)"],
                    ["а посулов бояром не [MASK]", "⚖️ Закон (Суд/Право)"],
                    ["во имѧ [MASK] и сн҃а", "⛪ Церковь (Библия)"],
                    ["и бысть въ [MASK] лѣто", "📖 Книжность (Летопись)"],
                    ["а [MASK] возми у него", "⚖️ Закон (Суд/Право)"],
                ],
                inputs=[input_text, style_radio],
            )

        with gr.Column():

            output_label = gr.Label(num_top_classes=5, label="Топ-5 предсказаний")
            output_text = gr.Textbox(label="Детали", interactive=False)

    btn.click(
        fn=predict_old_russian,
        inputs=[input_text, style_radio],
        outputs=[output_label, output_text],
    )


demo.launch(share=True, debug=True)
