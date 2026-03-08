import os
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast

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
    ]

    print(f"🧠 Начинаем обучение токенизатора на файле {CORPUS_FILE}...")
    print(f"Размер словаря: {VOCAB_SIZE} токенов")

    bpe_tokenizer.train(
        files=[CORPUS_FILE],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens,
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    bpe_tokenizer.save_model(SAVE_DIR)

    print("\n" + "=" * 50)
    print("✨ ТОКЕНИЗАТОР УСПЕШНО ОБУЧЕН И СОХРАНЕН ✨")
    print(f"Папка с файлами: {SAVE_DIR}/")
    print(
        "Внутри лежат файлы 'vocab.json' (словарь) и 'merges.txt' (правила слияния байтов)"
    )
    print("=" * 50)

    print("\n🔍 ТЕСТИРОВАНИЕ:")
    test_texts = [
        "[CTX_DAILY] Поклонъ ѿ бориса [GAP] настасии съ бг҃омъ.",
        "[CTX_SCIENCE] И царь самодержецъ повелѣлъ шанцы копати.",
        "[CTX_CHURCH] Преподобноисповѣдникъ моляшеся непрестанно.",
    ]

    # Загружаем через интерфейс Transformers (clean_up_tokenization_spaces убирает варнинг)
    fast_tokenizer = RobertaTokenizerFast.from_pretrained(
        SAVE_DIR, max_len=512, clean_up_tokenization_spaces=True
    )

    # 🔥 ВАЖНО: Явно сообщаем обертке Hugging Face о наших неделимых тегах
    special_tokens_dict = {
        "additional_special_tokens": [
            "[CTX_CHURCH]",
            "[CTX_DAILY]",
            "[CTX_LEGAL]",
            "[CTX_LIT]",
            "[CTX_EPIC]",
            "[CTX_SCIENCE]",
            "[GAP]",
        ]
    }
    fast_tokenizer.add_special_tokens(special_tokens_dict)

    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Тест {i} ---")
        print(f"Оригинал: {text}")

        # Токенизация (разбивка на подслова)
        tokens = fast_tokenizer.tokenize(text)
        print(f"Токены:   {tokens}")

        # Кодирование и декодирование
        encoded_ids = fast_tokenizer.encode(text)
        decoded_text = fast_tokenizer.decode(encoded_ids)
        print(f"Декодинг: {decoded_text}")


if __name__ == "__main__":
    main()
