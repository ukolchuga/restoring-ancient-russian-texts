import os
import random
from tqdm import tqdm
from transformers import BertTokenizerFast

# Путь к твоему обученному токенизатору
TOKENIZER_DIR = "../ancient_rus_tokenizer"

SOURCES_CONFIG = [
    {
        "type": "file",
        "path": "birch_bark/gramoty_clean.txt",
        "tag": "[CTX_DAILY]",
        "weight": 40,
    },
    {
        "type": "file",
        "path": "bilini/ultimate_ancient_rus_corpus.txt",
        "tag": "[CTX_EPIC]",
        "weight": 5,
    },
    {
        "type": "file",
        "path": "bible/bible_full_clean.txt",
        "tag": "[CTX_CHURCH]",
        "weight": 1,
    },
    {
        "type": "folder",
        "path": "json_data/diacu_CHURCH",
        "tag": "[CTX_CHURCH]",
        "weight": 1,
    },
    {"type": "folder", "path": "json_data/diacu_LIT", "tag": "[CTX_LIT]", "weight": 2},
    {
        "type": "folder",
        "path": "json_data/diacu_DAILY",
        "tag": "[CTX_DAILY]",
        "weight": 15,
    },
    {
        "type": "folder",
        "path": "json_data/diacu_LEGAL",
        "tag": "[CTX_LEGAL]",
        "weight": 1,
    },
    {
        "type": "folder",
        "path": "torot/torot_CHURCH",
        "tag": "[CTX_CHURCH]",
        "weight": 1,
    },
    {"type": "folder", "path": "torot/torot_LIT", "tag": "[CTX_LIT]", "weight": 3},
    {"type": "folder", "path": "torot/torot_DAILY", "tag": "[CTX_DAILY]", "weight": 15},
    {"type": "folder", "path": "torot/torot_LEGAL", "tag": "[CTX_LEGAL]", "weight": 10},
    {
        "type": "folder",
        "path": "pushkinskij_texts/clean_texts/CHURCH",
        "tag": "[CTX_CHURCH]",
        "weight": 1,  # Много текста, вес небольшой
    },
    {
        "type": "folder",
        "path": "pushkinskij_texts/clean_texts/LIT",
        "tag": "[CTX_LIT]",
        "weight": 2,
    },
    {
        "type": "folder",
        "path": "pushkinskij_texts/clean_texts/DAILY",
        "tag": "[CTX_DAILY]",
        "weight": 15,  # Быт усиливаем!
    },
    {
        "type": "folder",
        "path": "pushkinskij_texts/clean_texts/LEGAL",
        "tag": "[CTX_LEGAL]",
        "weight": 5,
    },
    {
        "type": "folder",
        "path": "pushkinskij_texts/clean_texts/EPIC",
        "tag": "[CTX_EPIC]",
        "weight": 4,  # Сказки и эпос
    },
    {
        "type": "folder",
        "path": "pushkinskij_texts/clean_texts/SCIENCE",
        "tag": "[CTX_SCIENCE]",
        "weight": 10,  # Науки мало, поэтому вес высокий, чтобы модель её запомнила
    },
]

OUTPUT_FILE = "final_ancient_rus_dataset.txt"


def load_and_process_source(config, tokenizer):
    path = config["path"]
    src_type = config["type"]
    tag = config["tag"]
    weight = config["weight"]

    if not os.path.exists(path):
        print(f"⚠️ Внимание: {path} не найден! Пропускаем.")
        return [], 0

    unique_lines = set()
    if src_type == "file":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    unique_lines.add(clean_line)
    elif src_type == "folder":
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt")]
        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    clean_line = line.strip()
                    if clean_line:
                        unique_lines.add(clean_line)

    tagged_lines = []
    source_tokens = 0

    # Считаем токены только для уникальных строк ОДИН раз, потом умножим на вес
    print(f"🔢 Считаем токены для {path}...")
    for line in unique_lines:
        # Учитываем тег в подсчете
        full_text = f"{tag} {line}"
        # Быстрая токенизация без лишних надстроек
        tokens_count = len(tokenizer.encode(full_text, add_special_tokens=False))
        source_tokens += tokens_count
        tagged_lines.extend([full_text] * weight)

    total_tokens_with_weight = source_tokens * weight
    print(
        f"✅ {path}: {len(unique_lines)} строк -> {total_tokens_with_weight:,} токенов (с весом {weight})"
    )

    return tagged_lines, total_tokens_with_weight


def main():
    print("⏳ Загрузка токенизатора для анализа...")
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_DIR)

    print("🌟 Начинаем великое слияние и аудит токенов...")
    all_final_lines = []
    stats = {}

    for source_cfg in SOURCES_CONFIG:
        tag = source_cfg["tag"]
        # Распаковываем данные
        lines, token_count = load_and_process_source(source_cfg, tokenizer)
        all_final_lines.extend(lines)

        # Накапливаем статистику по тегам
        stats[tag] = stats.get(tag, 0) + token_count

    print("\n" + "=" * 50)
    print("📊 ИТОГОВАЯ СТАТИСТИКА ПО ТОКЕНАМ:")
    total_tokens = sum(stats.values())
    for tag, count in stats.items():
        percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"{tag:<15}: {count:>12,} токенов ({percentage:>5.2f}%)")

    print(f"\n🚀 ОБЩИЙ ОБЪЕМ ДАТАСЕТА: {total_tokens:,} токенов")
    print("=" * 50)

    print("🔀 Перемешиваем строки...")
    random.seed(42)
    random.shuffle(all_final_lines)

    print(f"💾 Сохраняем в {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(all_final_lines))

    print("\n✨ МЕГА-КОРПУС ГОТОВ ✨")


if __name__ == "__main__":
    main()
