from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("https://example.com")

elements = driver.find_elements(By.TAG_NAME, "h1")

for el in elements:
    print(el.text)

driver.quit()
