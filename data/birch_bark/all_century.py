import pandas as pd
import re

INPUT_FILE = 'gramoty_text_only_cleaned.csv'
OUTPUT_FILE = 'gramoty_train_fixed.csv'


def get_century_label(year):

    century = (year // 100) + 1
    half = 1 if (year % 100) < 50 else 2


    roman = {10: 'X', 11: 'XI', 12: 'XII', 13: 'XIII', 14: 'XIV', 15: 'XV'}
    cen_str = roman[century]

    return f"{cen_str}_{half}"



def parse_year_mean(date_str):
    if pd.isna(date_str): return None
    nums = re.findall(r'\d{3,4}', date_str)
    return int((int(nums[0]) + int(nums[1])) / 2)


df = pd.read_csv(INPUT_FILE)
df['year_average'] = df['date'].apply(parse_year_mean)
df['century'] = df['year_average'].apply(get_century_label)


final_df = df[[
    'original_text_spaced', 'century', 'city_slug', 'year_average'
]]
final_df.columns = ['text', 'century', 'city', 'year']

final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')