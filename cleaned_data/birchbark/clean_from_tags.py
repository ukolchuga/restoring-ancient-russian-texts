import re


def clean_tags(input_path, output_path):
    # Паттерн ищет:
    # ^\[        - символ '[' строго в начале строки
    # [^\]]+     - любые символы, кроме закрывающей скобки
    # \]         - закрывающую скобку
    # \s*        - любые пробельные символы после скобки (пробелы, табы)
    pattern = r'^\[[^\]]+\]\s*'

    with open(input_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            clean_line = re.sub(pattern, '', line)
            f_out.write(clean_line)


clean_tags('birchbark_final_cleaned_with_brackets.txt', 'birchbark_final_cleaned_with_brackets_no_tags.txt')