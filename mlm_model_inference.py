from transformers import pipeline

MODEL_ID = "AlexSychovUN/mini-bert-ancient-rus-v2"

print("🔍 Загрузка обученной модели для финального теста...")
fill_mask = pipeline(
    "fill-mask",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device=0,
)


final_tests = [
    {
        "category": "⛪️ [CTX_CHURCH] (Ожидаем: сына / отца / бога / духа)",
        "text": "[CTX_CHURCH] Во имя отца и [MASK] и святаго духа.",
    },
    {
        "category": "🏡 [CTX_DAILY] (Ожидаем: господину / брату / юрью)",
        "text": "[CTX_DAILY] Поклонъ ѿ бориса ко [MASK] съ бг҃омъ.",
    },
    {
        "category": "⚖️ [CTX_LEGAL] (Ожидаем: винити / судити / имати / дати)",
        "text": "[CTX_LEGAL] Аже оубиеть моужь мужа, то мьстити брату, а посулов не [MASK] .",
    },
    {
        "category": "📚 [CTX_LIT] (Ожидаем: словесы / дѣлы)",
        "text": "[CTX_LIT] Не лѣпо ли ны бяшетъ братие начяти старыми [MASK] трудную повѣсть.",
    },
    {
        "category": "⚔️ [CTX_EPIC] (Ожидаем: молодец / богатырь / конь)",
        "text": "[CTX_EPIC] Гой еси ты добрый [MASK] , куда путь держишь?",
    },
    {
        "category": "🌿 [CTX_SCIENCE] (Ожидаем: зеліе / траву / воду)",
        "text": "[CTX_SCIENCE] А ѿ тоя болезни дай ему пити [MASK] , и тако исцелеет.",
    },
]

print("\n" + "=" * 60)
print("Final test for Mini Bert (All categories)")
print("=" * 60)

for test in final_tests:
    print(f"\n🔹 {test['category']}")
    print(f"Текст: {test['text']}")
    results = fill_mask(test["text"])
    for i, res in enumerate(results[:3]):
        print(f"  {i+1}. {res['token_str']:<12} (Уверенность: {res['score']*100:.1f}%)")
