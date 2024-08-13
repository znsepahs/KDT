from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# User Agent 정보 추가
agent_option= webdriver.ChromeOptions()
user_agent_srting='Mozilla/5.0'
agent_option.add_argument('user-agent=' + user_agent_srting)

driver=webdriver.Chrome(options=agent_option)
driver.get('http://nid.naver.com/nidlogin.login')

# <input>의 이름이 id를 검색
driver.find_element(By.NAME, 'id').send_keys('안알려줌')
driver.find_element(By.NAME, 'pw').send_keys('매우중요')

driver.find_element(By.ID, 'log.login').click()
time.sleep(3)
driver.quit()