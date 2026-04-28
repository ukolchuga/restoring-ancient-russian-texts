import json

# ========================================================================
# СМОТРИМ КАКИЕ ОТЛИЧИТЕЛЬНЫЕ СИМВОЛЫ ЕСТЬ В ДАТАСЕТЕ ГРАМОТ И ЭПИГРАФИКИ
# ========================================================================

chars_a = set()

with open('../data/splits/test_b.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        original_text = obj.get('original', '')
        chars_a.update(original_text)

with open('../data/splits/train.txt', 'r', encoding='utf-8') as f:
    text_b = f.read()

chars_b = set(text_b)

# разница
diff = chars_a - chars_b

print(sorted(diff))

# ========================================================================
# СМОТРИМ ПРИМЕРЫ НЕИЗВЕСТНЫХ СИМВОЛОВ
# ========================================================================

# список неизвестных символов
target_chars = [
    '\uf067', '\uf074', '\uf080', '\uf130', '\uf13f', '\uf147',
    '\uf14e', '\uf222', '\uf23a', '\uf245', '\uf265', '\uf27a', '\uf2b4',
    '\uf2b5', '\uf2c8', '\uf2d1', '\uf2db', '\uf42e', '\uf467', '\uf47e',
    '\uf480', '\uf488', '\uf48e', '\uf4a4', '\uf4a5'
]

# сюда будем складывать примеры
examples = {}

# сколько символов показывать вокруг
window = 30

with open('../data/splits/test_b.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        text = obj.get('original', '')

        for ch in target_chars:
            if ch in examples:
                continue

            idx = text.find(ch)
            if idx != -1:
                start = max(0, idx - window)
                end = idx + window

                context = text[start:end]

                examples[ch] = context

        if len(examples) == len(target_chars):
            break

# вывод
for ch in target_chars:
    print(f"{repr(ch)}:")
    print(examples.get(ch, "НЕ НАЙДЕНО"))
    print("-" * 50)


# на более раннем этапе, ‐ —> [GAP] — пофиксил руками

# нужно, чтобы знак титла ҃ сохранялся и в train, и в test

# +, :, · — добавить в список особых токенов наряду с MASK, GAP и проч.

# †, × —> +
# ⁘, ⁙, ⁞, ¦ —> :
# ∙, *, ., ҂, \uf13f —> ·
# ҇, ҃, \uf222, \uf23a, \uf2b4, \uf2b5 —> ҃
# ⃝, ⟦, ⟧, /, \\, |, ?, ;, ',', '̇', '̈', '̴', '͘', '', \u200e, \uf074, \uf080, \uf245, \uf265, \uf27a, \uf2db, \uf4a4, \uf4a5 —> удалить
# \uf074 —> ꙅ
# \uf130, \uf48e —> ꙩ
# \uf147 —> ѡ
# \uf14e, \uf42e —> ѿ
# \uf467 —> ѯ
# \uf47e, \uf480 —> ꙋ