from pathlib import Path

def check_overlap(train_path: str | Path, test_a_path: str | Path):
    """Проверяет пересечения между train и test_a."""

    train_path = Path(train_path)
    test_a_path = Path(test_a_path)

    print("=" * 80)
    print("Checking overlap between train.txt and test_a.txt")
    print("=" * 80)

    # Загружаем train
    print("\nLoading train.txt...")
    train_lines = set()
    train_count = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                train_lines.add(line)
                train_count += 1

    print(f"  Total lines: {train_count:,}")
    print(f"  Unique lines: {len(train_lines):,}")

    # Загружаем test_a
    print("\nLoading test_a.txt...")
    test_a_lines = set()
    test_a_count = 0
    with open(test_a_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_a_lines.add(line)
                test_a_count += 1

    print(f"  Total lines: {test_a_count:,}")
    print(f"  Unique lines: {len(test_a_lines):,}")

    # Проверяем пересечение
    overlap = train_lines & test_a_lines

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Overlap found: {len(overlap):,} lines")

    if overlap:
        print("\n⚠️  WARNING: Found duplicates!")
        print("\nFirst 10 overlapping lines:")
        for i, line in enumerate(sorted(overlap)[:10]):
            print(f"  {i+1}. {line[:100]}")

        # Сохраняем пересечение в файл
        overlap_path = train_path.parent / "overlap.txt"
        with open(overlap_path, "w", encoding="utf-8") as f:
            for line in sorted(overlap):
                f.write(line + "\n")
        print(f"\nOverlap saved to: {overlap_path}")
    else:
        print("\n✅ OK: No overlaps found!")

    print("\n" + "=" * 80)
    print(f"Train unique lines: {len(train_lines):,}")
    print(f"Test_a unique lines: {len(test_a_lines):,}")
    print(f"Total unique: {len(train_lines | test_a_lines):,}")
    print("=" * 80)

    return len(overlap) == 0


if __name__ == "__main__":
    import sys

    # Пути по умолчанию
    train_path = "splits/train.txt"
    test_a_path = "splits/test_a.txt"

    # Можно передать свои пути как аргументы
    if len(sys.argv) > 2:
        train_path = sys.argv[1]
        test_a_path = sys.argv[2]

    is_clean = check_overlap(train_path, test_a_path)
    sys.exit(0 if is_clean else 1)