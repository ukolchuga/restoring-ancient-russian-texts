import os

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

CORPUS_FILE = "data/ancient_rus_ready_for_bert.txt"
VOCAB_SIZE = 50000
SAVE_DIR = "ancient_rus_tokenizer_BPE"


def main():
    if not os.path.exists(CORPUS_FILE):
        print(f"❌ Ошибка: Файл {CORPUS_FILE} не найден!")
        return

    print("⏳ Инициализация BPE токенизатора...")
    bpe_tokenizer = ByteLevelBPETokenizer()

    special_tokens = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
        "[CTX_CHURCH]",
        "[CTX_DAILY]",
        "[CTX_LEGAL]",
        "[CTX_LIT]",
        "[CTX_EPIC]",
        "[CTX_SCIENCE]",
        "[GAP]",
        "·",
        ":",
    ]

    print(f"🧠 Обучение на {CORPUS_FILE} (vocab size: {VOCAB_SIZE})...")
    bpe_tokenizer.train(
        files=[CORPUS_FILE],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens,
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 🔥 ИСПРАВЛЕНИЕ 1: Сохраняем в единый монолитный формат tokenizer.json
    tokenizer_path = os.path.join(SAVE_DIR, "tokenizer.json")
    bpe_tokenizer.save(tokenizer_path)

    print("⏳ Настройка обертки Transformers...")
    # 🔥 ИСПРАВЛЕНИЕ 2: Используем универсальный PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        max_len=512,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    # Регистрируем теги
    fast_tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "[CTX_CHURCH]",
                "[CTX_DAILY]",
                "[CTX_LEGAL]",
                "[CTX_LIT]",
                "[CTX_EPIC]",
                "[CTX_SCIENCE]",
                "[GAP]",
                "·",
                ":",
            ]
        }
    )

    # Сохраняем HF конфиг
    fast_tokenizer.save_pretrained(SAVE_DIR)

    print("\n" + "=" * 50)
    print("✨ BPE ТОКЕНИЗАТОР УСПЕШНО ОБУЧЕН И СОХРАНЕН ✨")
    print(f"Путь: {SAVE_DIR}/")
    print("=" * 50)

    print("\n🔍 ТЕСТИРОВАНИЕ:")
    test_texts = [
        "[CTX_DAILY] поклонъ · ѿ · бориса · [GAP] · настасии",
        "[CTX_LEGAL] · ꙅ҃ · десѧ · коуно ·",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\nТест {i}: {text}")
        tokens = fast_tokenizer.tokenize(text)
        print(f"Токены: {tokens}")
        ids = fast_tokenizer.encode(text)
        print(f"Декодинг: {fast_tokenizer.decode(ids)}")


if __name__ == "__main__":
    main()
