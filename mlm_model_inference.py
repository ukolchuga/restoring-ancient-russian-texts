from transformers import pipeline

MODEL_PATH = "./old_rus_bert_best/checkpoint-3600"


fill_mask = pipeline(
    "fill-mask",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)

def test_model(text):
    print(f"\n📖 Text: {text}")
    results = fill_mask(text)
    for res in results[:3]:
        print(f"   {res['score']:.1%} -> {res['token_str']}")


texts = [
"поклоно ѿ онѳима ко [MASK]",
"а посулов бояром и околничим не [MASK]",
"за млтвѹ стхъ ѡць наших ги їсе хе сне бжїи [MASK] мѧ",
"во имѧ ѿц҃а и [MASK] и ст҃го дх҃а",
"господи [MASK] мѧ грѣшника"
]
for text in texts:
    test_model(text)