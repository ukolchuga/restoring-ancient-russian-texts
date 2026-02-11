import pandas as pd


data_xlsx = pd.read_excel("ruscorpora_content.xlsx")


def get_historical_period(year):
    century = (year // 100) + 1

    half = 1 if (year % 100) < 50 else 2
    roman_numerals = {
        10: "X",
        11: "XI",
        12: "XII",
        13: "XIII",
        14: "XIV",
        15: "XV",
        16: "XVI",
        17: "XVII",
    }
    century_roman = roman_numerals[century]
    return f"{century_roman}c. {half} h."


def parse_years(year: str):
    year1, year2 = year.split("-")
    year1 = int(year1)
    year2 = int(year2)

    avg_year = (year1 + year2) // 2
    return year1, year2, avg_year


def parse_gramoty(data: pd.DataFrame):
    documents = []

    # Aggregation
    grouped = (
        data.groupby("Header")
        .agg(
            {
                "Full context": "first",
                "Created": "first",
                "Title": "first",
                "Para context 1": "first",  # Russian
                "Para context 2": "first",  # Eng1
                "Para context 3": "first",  # Eng2
            }
        )
        .reset_index()
    )

    years = grouped["Created"].apply(parse_years)
    grouped["year_start"] = years.apply(lambda x: x[0])
    grouped["year_end"] = years.apply(lambda x: x[1])
    grouped["target_year"] = years.apply(lambda x: x[2])

    grouped["time_period"] = grouped["target_year"].apply(get_historical_period)

    dataset = grouped[
        [
            "Header",
            "Full context",
            "time_period",
            "target_year",
            "year_start",
            "year_end",
            "Para context 1",
            "Para context 2",
            "Para context 3",
        ]
    ]
    dataset.columns = [
        "doc_id",
        "text",
        "time_period",
        "avg_year",
        "year_start",
        "year_end",
        "translation_ru",
        "translation_en_1",
        "translation_en_2",
    ]
    print(f"Gathered {len(dataset)} documents.")
    print(f"\nExample of data:")
    print(dataset[["doc_id", "time_period", "text"]].head(5))
    dataset.to_csv("gramoty.csv", index=False, encoding="utf-8-sig")


parse_gramoty(data_xlsx)
