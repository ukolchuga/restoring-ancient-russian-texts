import os
import time
import random
from bs4 import BeautifulSoup
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# 1. Структура папок по подкорпусам НКРЯ
BASE_DIR = "NKRYA_RAW"
CORPORA = [
    "Birch",
    "Epigraph",
    "Old_Rus",
    "Mid_Rus",
    "Church",
    "SCIENCE",
    "LEGAL",
    "DAILY",
    "LIT",
]

# Создаем базовые папки
for corp in CORPORA:
    os.makedirs(os.path.join(BASE_DIR, corp), exist_ok=True)


def setup_driver():
    options = Options()
    options.add_argument(
        "--headless"
    )  # Раскомментируй, когда отладишь, чтобы браузер работал в фоне
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def scrape_subcorpus(
    driver, url, corpus_name, source_name, max_pages=100, start_page=1
):
    # Создаем подпапку для конкретного источника/ссылки
    source_dir = os.path.join(BASE_DIR, corpus_name, source_name)
    os.makedirs(source_dir, exist_ok=True)

    print(
        f"\n🚀 Выкачиваем: [{corpus_name}] -> {source_name} (старт со страницы {start_page})"
    )

    driver.set_page_load_timeout(60)

    # Загружаем стартовую страницу
    max_retries = 3
    for attempt in range(max_retries):
        try:
            driver.get(url)
            break
        except Exception as e:
            print(f"⚠️ Ошибка сети (попытка {attempt+1}/{max_retries}). Ждем 10 сек...")
            time.sleep(10)
    else:
        print(f"❌ Не удалось пробиться на сайт для {source_name}. Пропускаем.")
        return

    # ПРОМОТКА ДО НУЖНОЙ СТРАНИЦЫ (Если запуск упал и мы возобновляем)
    if start_page > 1:
        print(f"⏩ Быстрая промотка до страницы {start_page}...")
        for _ in tqdm(range(start_page - 1), desc="Промотка"):
            try:
                next_btn = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located(
                        (
                            By.XPATH,
                            "//button[@aria-label='Следующая страница'] | //li[contains(@class, 'next')]/a | //button[contains(@class, 'pager__next')]",
                        )
                    )
                )
                driver.execute_script("arguments[0].click();", next_btn)
                time.sleep(1.5)
            except Exception as e:
                print(
                    "⚠️ Ошибка при промотке! Сервер не отдал кнопку. Начнем парсинг отсюда."
                )
                break

    collected_texts = set()
    timestamp = int(time.time())
    global_idx = 0

    # Основной цикл сбора
    for page in tqdm(
        range(start_page, max_pages + 1),
        desc=f"Сбор ({source_name})",
        initial=start_page - 1,
        total=max_pages,
    ):
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        "li.concordance-item, p.concordance-sequence, p.seq-with-actions",
                    )
                )
            )

            soup = BeautifulSoup(driver.page_source, "html.parser")
            # Метод select позволяет искать сразу по нескольким CSS-селекторам через запятую
            paragraphs = soup.select("p.concordance-sequence, p.seq-with-actions")

            if not paragraphs:
                raise ValueError("Текст не прогрузился (пустая страница).")

            for p in paragraphs:
                text = p.get_text(separator=" ", strip=True)
                text = " ".join(text.split())

                # === СОХРАНЕНИЕ НА ЛЕТУ В ПАПКУ ИСТОЧНИКА ===
                if len(text) > 15 and text not in collected_texts:
                    collected_texts.add(text)

                    file_name = f"record_{timestamp}_{global_idx:05d}.txt"
                    output_path = os.path.join(source_dir, file_name)

                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(text)

                    global_idx += 1
                # ==========================

            next_btn = driver.find_elements(
                By.XPATH,
                "//button[@aria-label='Следующая страница'] | //li[contains(@class, 'next')]/a | //button[contains(@class, 'pager__next')]",
            )

            if not next_btn or not next_btn[0].is_enabled():
                print(f"\n🏁 Последняя страница достигнута на странице {page}.")
                break

            time.sleep(random.uniform(3.5, 6.5))
            driver.execute_script("arguments[0].click();", next_btn[0])
            time.sleep(random.uniform(1.5, 3.0))

        except Exception as e:
            print(f"\n❌ СБОЙ НА СТРАНИЦЕ {page} ИСТОЧНИКА {source_name}.")
            print(f"Причина: {str(e)[:100]}")
            print(
                f"💡 Чтобы продолжить, обнови задачу: ('URL', '{corpus_name}', '{source_name}', {max_pages}, {page})"
            )

            screenshot_name = os.path.join(
                source_dir, f"error_page_{page}_{int(time.time())}.png"
            )
            driver.save_screenshot(screenshot_name)
            break

    print(f"✅ Успешно сохранено {global_idx} новых файлов в папку {source_name}")


if __name__ == "__main__":
    driver = setup_driver()

    try:
        # ЗАДАЧА:
        # 4 аргумента: (URL, ТЕГ, ИМЯ_ИСТОЧНИКА, СТРАНИЦЫ)
        # Если нужно резюмировать, добавляем 5-й аргумент: (..., СТРАНИЦЫ, СТАРТОВАЯ_СТРАНИЦА)

        tasks = [
            # === НАУКА (SCIENCE) ===
            # (
            #     "https://ruscorpora.ru/results?search=CpsCEnkKdwoYCghvcnRob21vZBIMCgpzaW1wbGlmaWVkChcKB2Rpc3Rtb2QSDAoKd2l0aF96ZXJvcxJCCgkKA2xleBICCgAKDwoJbGV4X2Vhcmx5EgIKAAoKCgRmb3JtEgIKAAoLCgVncmFtbRICCgAKCwoFZmxhZ3MSAgoAImwKagpHCghjYXRlZ29yeRI7CjnQvdCw0YPRh9C90YvQtSDRgtC10LrRgdGC0YsgfCDRg9GH0LXQsdC90YvQtSDRgtC10LrRgdGC0YsKHwoHY3JlYXRlZDIUCgcI%252BAoQARgBEgcIpA0QDBgfGAIqKQoICAAQChgyIAogACissvPswZX4ATIFZ3JzdGRABWoEMC45NXgBoAEBMgIIEToBAQ%253D%253D",
            #     "SCIENCE",
            #     "starorus_science_1400_1700",
            #     1,
            # ),
            # (
            #     "https://ruscorpora.ru/results?search=CpoCEpUBCpIBChgKCG9ydGhvbW9kEgwKCnNpbXBsaWZpZWQKFwoHZGlzdG1vZBIMCgp3aXRoX3plcm9zEl0KCQoDbGV4EgIKAAoKCgRmb3JtEgIKAAoLCgVncmFtbRICCgAKCwoFZmxhZ3MSAgoACg0KB21lYW5pbmcSAgoACgwKBmxleF9lbBICCgAKDQoHZm9ybV9lbBICCgAiTgpMCikKCGNhdGVnb3J5Eh0KG9C90LDRg9GH0L3Ri9C1INGC0LXQutGB0YLRiwofCgdjcmVhdGVkMhQKBwjoBxABGAESBwj4ChAMGB8YAiopCggIABAKGDIgCiAAKKyy8%252BzBlfgBMgVncnN0ZEAFagQwLjk1eAGgAQEyAggPOgEB",
            #     "SCIENCE",
            #     "oldrus_science_1000_1400",
            #     1,
            # ),
            # (
            #     "https://ruscorpora.ru/results?search=CvUBEpEBCo4BChgKCG9ydGhvbW9kEgwKCnNpbXBsaWZpZWQKFAoLbGFuZ19zZWFyY2gSBQoDb3J2ElwKCQoDbGV4EgIKAAoPCglsZXhfZWFybHkSAgoACgoKBGZvcm0SAgoACgsKBWdyYW1tEgIKAAoNCgdtZWFuaW5nEgIKAAoJCgNzZW0SAgoACgsKBWZsYWdzEgIKACItCisKKQoIY2F0ZWdvcnkSHQob0YPRh9C10LHQvdGL0LUg0YLQtdC60YHRgtGLKikKCAgAEAoYMiAKIAAorLLz7MGV%252BAEyBWdyc3RkQAVqBDAuOTV4AaABATICCBA6AQE%253D",
            #     "SCIENCE",
            #     "birch_science_edu",
            #     3,
            # ),
            # # === ЗАКОНЫ (LEGAL) ===
            # (
            #     "https://ruscorpora.ru/results?search=CvsBEnkKdwoYCghvcnRob21vZBIMCgpzaW1wbGlmaWVkChcKB2Rpc3Rtb2QSDAoKd2l0aF96ZXJvcxJCCgkKA2xleBICCgAKDwoJbGV4X2Vhcmx5EgIKAAoKCgRmb3JtEgIKAAoLCgVncmFtbRICCgAKCwoFZmxhZ3MSAgoAIlwKWgo3CghjYXRlZ29yeRIrCinQvtGE0LjRhtC40LDQu9GM0L3Ri9C1INC00L7QutGD0LzQtdC90YLRiwofCgdjcmVhdGVkMhQKBwj4ChABGAESBwjcCxAMGB8YAioZCggIABAKGDIgCiAAQAVqBDAuOTV4AaABATICCBE6AQE%253D",
            #     "LEGAL",
            #     "starorus_legal_1400_1500",
            #     100,
            # ),
            # (
            #     "https://ruscorpora.ru/results?search=CvsBEnkKdwoYCghvcnRob21vZBIMCgpzaW1wbGlmaWVkChcKB2Rpc3Rtb2QSDAoKd2l0aF96ZXJvcxJCCgkKA2xleBICCgAKDwoJbGV4X2Vhcmx5EgIKAAoKCgRmb3JtEgIKAAoLCgVncmFtbRICCgAKCwoFZmxhZ3MSAgoAIlwKWgo3CghjYXRlZ29yeRIrCinQvtGE0LjRhtC40LDQu9GM0L3Ri9C1INC00L7QutGD0LzQtdC90YLRiwofCgdjcmVhdGVkMhQKBwjdCxABGAESBwiODBAMGB8YAioZCggIABAKGDIgCiAAQAVqBDAuOTV4AaABATICCBE6AQE%253D",
            #     "LEGAL",
            #     "starorus_legal_1501_1550",
            #     55,
            # ),
            # (
            #     "https://ruscorpora.ru/results?search=CvsBEnkKdwoYCghvcnRob21vZBIMCgpzaW1wbGlmaWVkChcKB2Rpc3Rtb2QSDAoKd2l0aF96ZXJvcxJCCgkKA2xleBICCgAKDwoJbGV4X2Vhcmx5EgIKAAoKCgRmb3JtEgIKAAoLCgVncmFtbRICCgAKCwoFZmxhZ3MSAgoAIlwKWgo3CghjYXRlZ29yeRIrCinQvtGE0LjRhtC40LDQu9GM0L3Ri9C1INC00L7QutGD0LzQtdC90YLRiwofCgdjcmVhdGVkMhQKBwiPDBABGAESBwisDBAMGB8YAioZCggIABAKGDIgCiAAQAVqBDAuOTV4AaABATICCBE6AQE%253D",
            #     "LEGAL",
            #     "starorus_legal_1551_1580",
            #     73,
            # ),
            # (
            #     "https://ruscorpora.ru/results?search=CvsBEnkKdwoYCghvcnRob21vZBIMCgpzaW1wbGlmaWVkChcKB2Rpc3Rtb2QSDAoKd2l0aF96ZXJvcxJCCgkKA2xleBICCgAKDwoJbGV4X2Vhcmx5EgIKAAoKCgRmb3JtEgIKAAoLCgVncmFtbRICCgAKCwoFZmxhZ3MSAgoAIlwKWgo3CghjYXRlZ29yeRIrCinQvtGE0LjRhtC40LDQu9GM0L3Ri9C1INC00L7QutGD0LzQtdC90YLRiwofCgdjcmVhdGVkMhQKBwitDBABGAESBwjADBAMGB8YAioZCggIABAKGDIgCiAAQAVqBDAuOTV4AaABATICCBE6AQE%253D",
            #     "LEGAL",
            #     "starorus_legal_1581_1600",
            #     71,
            # ),
            # # === БЫТ (DAILY) ===
            # (
            #     "https://ruscorpora.ru/results?search=CowCEnkKdwoYCghvcnRob21vZBIMCgpzaW1wbGlmaWVkChcKB2Rpc3Rtb2QSDAoKd2l0aF96ZXJvcxJCCgkKA2xleBICCgAKDwoJbGV4X2Vhcmx5EgIKAAoKCgRmb3JtEgIKAAoLCgVncmFtbRICCgAKCwoFZmxhZ3MSAgoAIl0KWwo4CghjYXRlZ29yeRIsCirQtNC10LvQvtCy0YvQtSDQt9Cw0L%252FQuNGB0LggfCDQv9C40YHRjNC80LAKHwoHY3JlYXRlZDIUCgcI%252BAoQARgBEgcI8gwQDBgfGAIqKQoICAAQChgyIAogACissvPswZX4ATIFZ3JzdGRABWoEMC45NXgBoAEBMgIIEToBAQ%253D%253D",
            #     "DAILY",
            #     "starorus_daily_1400_1650",
            #     54,
            # ),
            # (
            #     "https://ruscorpora.ru/results?search=CowCEnkKdwoYCghvcnRob21vZBIMCgpzaW1wbGlmaWVkChcKB2Rpc3Rtb2QSDAoKd2l0aF96ZXJvcxJCCgkKA2xleBICCgAKDwoJbGV4X2Vhcmx5EgIKAAoKCgRmb3JtEgIKAAoLCgVncmFtbRICCgAKCwoFZmxhZ3MSAgoAIl0KWwo4CghjYXRlZ29yeRIsCirQtNC10LvQvtCy0YvQtSDQt9Cw0L%252FQuNGB0LggfCDQv9C40YHRjNC80LAKHwoHY3JlYXRlZDIUCgcI8wwQARgBEgcIpA0QDBgfGAIqKQoICAAQChgyIAogACissvPswZX4ATIFZ3JzdGRABWoEMC45NXgBoAEBMgIIEToBAQ%253D%253D",
            #     "DAILY",
            #     "starorus_daily_1651_1700",
            #     148,
            # ),
            # (
            #     "https://ruscorpora.ru/results?search=CoQCEpEBCo4BChgKCG9ydGhvbW9kEgwKCnNpbXBsaWZpZWQKFAoLbGFuZ19zZWFyY2gSBQoDb3J2ElwKCQoDbGV4EgIKAAoPCglsZXhfZWFybHkSAgoACgoKBGZvcm0SAgoACgsKBWdyYW1tEgIKAAoNCgdtZWFuaW5nEgIKAAoJCgNzZW0SAgoACgsKBWZsYWdzEgIKACI8CjoKOAoIY2F0ZWdvcnkSLAoq0LTQtdC70L7QstGL0LUg0LfQsNC%252F0LjRgdC4IHwg0L%252FQuNGB0YzQvNCwKikKCAgAEAoYMiAKIAAorLLz7MGV%252BAEyBWdyc3RkQAVqBDAuOTV4AaABATICCBA6AQE%253D",
            #     "DAILY",
            #     "birch_daily_letters",
            #     95,
            # ),
            # (
            #     "https://ruscorpora.ru/results?search=CuQCEpABCo0BChgKCG9ydGhvbW9kEgwKCnNpbXBsaWZpZWQKFAoLbGFuZ19zZWFyY2gSBQoDb3J2ElsKCQoDbGV4EgIKAAoOCghsZXhfbGF0ZRICCgAKCgoEZm9ybRICCgAKCwoFZ3JhbW0SAgoACg0KB21lYW5pbmcSAgoACgkKA3NlbRICCgAKCwoFZmxhZ3MSAgoAIpwBCpkBCpYBCghjYXRlZ29yeRKJAQqGAdC00LXQu9C%252B0LLRi9C1INC90LDQtNC%252F0LjRgdC4IHwg0LjQvNC10L3QvdGL0LUg0L3QsNC00L%252FQuNGB0LggfCDQv9Cw0LzRj9GC0L3Ri9C1INC90LDQtNC%252F0LjRgdC4IHwg0YPQv9GA0LDQttC90LXQvdC40Y8g0LIg0L%252FQuNGB0YzQvNC1KikKCAgAEAoYMiAKIAAorLLz7MGV%252BAEyBWdyc3RkQAVqBDAuOTV4AaABATICCBc6AQE%253D",
            #     "DAILY",
            #     "epigraph_daily_records",
            #     29,
            # ),
            # === ЛИТЕРАТУРА (LIT) ===
            # (
            #     "https://ruscorpora.ru/results?search=CqECEnkKdwoYCghvcnRob21vZBIMCgpzaW1wbGlmaWVkChcKB2Rpc3Rtb2QSDAoKd2l0aF96ZXJvcxJCCgkKA2xleBICCgAKDwoJbGV4X2Vhcmx5EgIKAAoKCgRmb3JtEgIKAAoLCgVncmFtbRICCgAKCwoFZmxhZ3MSAgoAInIKcApNCghjYXRlZ29yeRJBCj%252FQu9C40YLQtdGA0LDRgtGD0YDQvdGL0LUg0Lgg0YTQvtC70YzQutC70L7RgNC90YvQtSDRgtC10LrRgdGC0YsKHwoHY3JlYXRlZDIUCgcI%252BAoQARgBEgcIpA0QDBgfGAIqKQoICAAQChgyIAogACissvPswZX4ATIFZ3JzdGRABWoEMC45NXgBoAEBMgIIEToBAQ%253D%253D",
            #     "LIT",
            #     "starorus_lit_folklore_1400_1700",
            #     57,
            # ),
            (
                "https://ruscorpora.ru/results?search=CvkBEpUBCpIBChgKCG9ydGhvbW9kEgwKCnNpbXBsaWZpZWQKFwoHZGlzdG1vZBIMCgp3aXRoX3plcm9zEl0KCQoDbGV4EgIKAAoKCgRmb3JtEgIKAAoLCgVncmFtbRICCgAKCwoFZmxhZ3MSAgoACg0KB21lYW5pbmcSAgoACgwKBmxleF9lbBICCgAKDQoHZm9ybV9lbBICCgAiLQorCikKCGNhdGVnb3J5Eh0KG9Cx0YvRgtC%252B0LLRi9C1INC30LDQv9C40YHQuCopCggIABAKGDIgCiAAKKyy8%252BzBlfgBMgVncnN0ZEAFagQwLjk1eAGgAQEyAggPOgEB",
                "DAILY",
                "drevnirusskiy_bit",
                3,
            ),
            (
                "https://ruscorpora.ru/results?search=Cp0CEpUBCpIBChgKCG9ydGhvbW9kEgwKCnNpbXBsaWZpZWQKFwoHZGlzdG1vZBIMCgp3aXRoX3plcm9zEl0KCQoDbGV4EgIKAAoKCgRmb3JtEgIKAAoLCgVncmFtbRICCgAKCwoFZmxhZ3MSAgoACg0KB21lYW5pbmcSAgoACgwKBmxleF9lbBICCgAKDQoHZm9ybV9lbBICCgAiUQpPCk0KCGNhdGVnb3J5EkEKP9C70LjRgtC10YDQsNGC0YPRgNC90YvQtSDQuCDRhNC%252B0LvRjNC60LvQvtGA0L3Ri9C1INGC0LXQutGB0YLRiyopCggIABAKGDIgCiAAKKyy8%252BzBlfgBMgVncnN0ZEAFagQwLjk1eAGgAQEyAggPOgEB",
                "LIT",
                "drevnirusskiy_literature_folklore",
                8,
            ),
        ]

        # Теперь мы распаковываем все параметры из кортежа (включая start_page, если он передан)
        for task in tasks:
            scrape_subcorpus(driver, *task)
            time.sleep(10)

    finally:
        driver.quit()
        print("🛑 Работа парсера завершена.")
