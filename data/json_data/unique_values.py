import collections
import json

JSON_FILE = "DIACU_1.0.json"


def get_unique_metadata():
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = data.get("Documents", [])

        # Словари для сбора уникальных значений
        stats = {
            "Century": collections.Counter(),
            "Language": collections.Counter(),
            "Epoch": collections.Counter(),
        }

        for doc in docs:
            for field in stats.keys():
                val = doc.get(field, "ПУСТО").strip()
                stats[field][val] += 1

        for field, counter in stats.items():
            print(f"\n=== УНИКАЛЬНЫЕ ЗНАЧЕНИЯ: {field.upper()} ===")
            # Выводим топ-20, если их слишком много
            for val, count in counter.most_common(100):
                print(f"[{count:4d}] {val}")

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    get_unique_metadata()
