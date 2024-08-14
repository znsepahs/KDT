from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import json
from selenium import webdriver
from selenium.webdriver.common.by import By

url="https://careerly.co.kr/qnas/tagged/react"
driver = webdriver.Chrome()  # 본인의 webdriver 경로
driver.get(url)

xpath1_title =       '//*[@id="__next"]/div/div[2]/div/div/a[1]/p[1]/span'
title1_text = driver.find_element(By.XPATH, xpath1_title).text
print(title1_text)

xpath1_content = '//*[@id="__next"]/div/div[2]/div/div/a[1]/p[2]'
content1_text = driver.find_element(By.XPATH, xpath1_content).text
print(content1_text)

xpath2_title= '//*[@id="__next"]/div/div[2]/div/div/a[2]/p[1]/span'
title2_text = driver.find_element(By.XPATH, xpath2_title).text
print(title2_text)

driver.quit()
