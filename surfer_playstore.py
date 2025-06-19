from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

chrome_options = Options()
# chrome_options.add_argument("--headless")
# chrome_options.add_argument("--disable-gpu")
# chrome_options.add_argument("--window-size=1920x1080")

driver = None

app_ids = []

# u can change this app list
with open("app_list_2.txt","r") as file:
    app_ids = [line.strip() for line in file if line.strip()]




try:
    for APP_ID in app_ids:
        print(f"-----------Fetching reviews for {APP_ID}-----------")
        driver = webdriver.Chrome(options=chrome_options)
        PAGINATION = 60
        url = f"https://play.google.com/store/apps/details?id={APP_ID}&hl=en&gl=US"
        try:
            driver.get(url)
            # Check for 404 in page title or content
            if "404" in driver.title or "Not Found" in driver.title or "not found" in driver.page_source.lower():
                print(f"404 Not Found for {APP_ID}, quitting driver.")
                driver.quit()
                continue
        except Exception as e:
            print(f"Error loading URL for {APP_ID}: {e}")
            driver.quit()
            continue

        time.sleep(1)

        for _ in range(3):
            try:
                driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
            except Exception as e:
                print("Error sending PAGE_DOWN:", e)

        try:
            see_all = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//span[contains(text(), "See all reviews")]'))
            )
            driver.execute_script("arguments[0].scrollIntoView();", see_all)
            driver.execute_script("arguments[0].click()", see_all)
            print("Opened see all reviews...")
            time.sleep(1)
        except Exception as e:
            print("Could not open 'See all reviews':", e)
            raise

        def select_star_option(star_count):
            try:
                time.sleep(1)
                option_xpath = f'//div[@role="menu"]//div[contains(text(), "{star_count}-star")]'
                star_option = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, option_xpath))
                )
                star_option.click()
                time.sleep(1)
            except Exception as e:
                print(f"Could not select star option {star_count}:", e)

        all_data = []
        # Fetching only 1-4 star reviews
        for star in range(1, 5):
            print(f"Fetching {star} star reviews....")
            try:
                time.sleep(1)
                try:
                    relevant_btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, '//div[@role="button" and .//span[contains(text(), "Most relevant")]]'))
                    )
                    relevant_btn.click()
                    time.sleep(0.5)
                    relevant_menu = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, '//div[@role="menu"]//div[contains(text(), "Newest")]'))
                    )
                    relevant_menu.click()
                    time.sleep(1)
                except Exception as e:
                    print("Could not set sorting to Newest:", e)

                try:
                    star_rating_btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, '//div[@role="button" and .//span[contains(text(), "Star rating")]]'))
                    )
                    star_rating_btn.click()
                    time.sleep(0.5)
                    select_star_option(star)
                except Exception as e:
                    print(f"Could not open/select star rating for {star}:", e)
                    continue

                for _ in range(PAGINATION):
                    try:
                        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
                        time.sleep(0.5)
                    except Exception as e:
                        print("Error sending END key:", e)

                try:
                    review_blocks = driver.find_elements(By.XPATH, '//div[@class="RHo1pe"]')
                    for block in review_blocks:
                        try:
                            review_text = block.find_element(By.CLASS_NAME, "h3YV2d").text
                            date_span = block.find_element(By.CLASS_NAME, "bp9Aid").text
                            all_data.append({'Review': review_text, 'Date': date_span, 'Stars': star})
                        except Exception as e:
                            print("Error parsing review:", e)
                except Exception as e:
                    print("Error finding review blocks:", e)

                try:
                    driver.execute_script("arguments[0].click()", see_all)
                except Exception as e:
                    print("Error clicking 'See all reviews' again:", e)

            except Exception as e:
                print(f"Error in star loop for star={star}: {e}")
                continue

        if all_data:
            df = pd.DataFrame(all_data)

            try:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                filtered_df = df[df['Date'] > '2024-01-01']
                filtered_df.to_csv(f'reviews_{APP_ID}.csv', index=False)
                print('Dumping data to file')
            except Exception as e:
                print("Error saving CSV:", e)
        else:
            print("No data collected.")

except Exception as e:
    print("Fatal error:", e)
finally:
    if driver:
        driver.quit()
