import os
from tokenizers import BertWordPieceTokenizer


CORPUS_FILE = "data/ancient_rus_ready_for_bert.txt"
VOCAB_SIZE = 50000
SAVE_DIR = "ancient_rus_tokenizer"


def main():
    if not os.path.exists(CORPUS_FILE):
        print(f"❌ Ошибка: Файл {CORPUS_FILE} не найден!")
        return

    print("⏳ Инициализация WordPiece токенизатора...")

    # clean_text=False -> мы уже сами всё идеально почистили
    # strip_accents=False -> ЗАПРЕЩАЕМ удалять титла!
    # lowercase=False -> мы уже сделали Sentence case (Первая буква заглавная)
    tokenizer = BertWordPieceTokenizer(
        clean_text=False,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
    )

    special_tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[GAP]",
        "[CTX_DAILY]",
        "[CTX_EPIC]",
        "[CTX_LIT]",
        "[CTX_LEGAL]",
        "[CTX_CHURCH]",
        "[CTX_SCIENCE]",
    ]

    print(f"🧠 Начинаем обучение токенизатора на файле {CORPUS_FILE}...")
    print(f"Размер словаря: {VOCAB_SIZE} токенов")

    tokenizer.train(
        files=[CORPUS_FILE],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens,
        wordpieces_prefix="##",
    )

    # Создаем папку и сохраняем
    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer.save_model(SAVE_DIR)

    print("\n" + "=" * 50)
    print("✨ ТОКЕНИЗАТОР УСПЕШНО ОБУЧЕН И СОХРАНЕН ✨")
    print(f"Папка с файлами: {SAVE_DIR}/")
    print("Внутри лежат файлы 'vocab.txt' (твой уникальный словарь)")
    print("=" * 50)

    print("\n🔍 ТЕСТИРОВАНИЕ:")
    test_texts = [
        "[CTX_DAILY] Поклонъ ѿ бориса [GAP] настасии съ бг҃омъ.",
        "[CTX_SCIENCE] И царь самодержецъ повелѣлъ шанцы копати.",
        "[CTX_CHURCH] Преподобноисповѣдникъ моляшеся непрестанно.",
    ]

    from transformers import BertTokenizerFast

    fast_tokenizer = BertTokenizerFast.from_pretrained(
        SAVE_DIR, strip_accents=False, lowercase=False
    )

    special_tokens_dict = {
        "additional_special_tokens": [
            "[CTX_DAILY]",
            "[CTX_EPIC]",
            "[CTX_LIT]",
            "[CTX_LEGAL]",
            "[CTX_CHURCH]",
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

        # Обрати внимание, что у WordPiece подслова начинаются с ##
        encoded_ids = fast_tokenizer.encode(text)
        decoded_text = fast_tokenizer.decode(encoded_ids)
        print(f"Декодинг: {decoded_text}")


if __name__ == "__main__":
    main()
