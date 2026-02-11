text_file = "final_dataset_clean.txt"

word_freq = {}

with open(text_file, "r", encoding="utf-8") as f:
    for line in f:
        words = line.split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

# Sorting the words by frequency
sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

print("Top 100 words by frequency:")
for word, freq in sorted_words[:100]:
    print(f"{word}: {freq}")
