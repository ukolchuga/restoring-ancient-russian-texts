import os
import re
import pandas as pd
from collections import defaultdict

folders = ["DAILY", "LEGAL", "SCIENCE", "LIT"]
# Регулярка: ловим слова (\w+) и любые знаки пунктуации ([^\w\s])
token_pattern = re.compile(r"\w+|[^\w\s]")

file_stats = []
line_map = defaultdict(list)

for folder in folders:
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                path = os.path.join(folder, filename)
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    content = "".join(lines)

                    # Считаем токены
                    tokens = token_pattern.findall(content)
                    file_stats.append(
                        {
                            "category": folder,
                            "filename": filename,
                            "tokens": len(tokens),
                        }
                    )

                    # Индексируем строки для поиска дублей
                    for line_no, line in enumerate(lines, 1):
                        clean_line = line.strip()
                        if clean_line:
                            line_map[clean_line].append(
                                f"{folder}/{filename} (строка {line_no})"
                            )

# 1. Сохраняем статистику в CSV
df = pd.DataFrame(file_stats)
df.to_csv("corpus_stats.csv", index=False)

# 2. Ищем дубликаты
duplicates = {line: locs for line, locs in line_map.items() if len(locs) > 1}

print("Статистика по файлам:")
print(df)
if duplicates:
    print(f"\nНайдено повторяющихся строк: {len(duplicates)}")
    # Выведем примеры дублей
    for line, locs in list(duplicates.items())[:5]:
        print(f"Строка: '{line[:50]}...'")
        print(f"Встречается в: {', '.join(locs)}")
