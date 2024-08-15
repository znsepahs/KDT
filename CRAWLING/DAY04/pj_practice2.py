#---------------------------------------------------------------------------------
# it 대표주 :삼성전자, 삼성에스디에스(윤호)
# PER, PBR, ROE, 영업이익증가율, 투자활동현금흐름
# 각자 회사 2개씩 맡아서 3년치 선 그래프(추이 분석) 그리기.
#---------------------------------------------------------------------------------
from urllib.request import Request
import re
import pandas as pd
from urllib.parse import quote
from urllib.request import urlopen
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import platform
import numpy as np
from PIL import Image
from selenium import webdriver
import collections

if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

#-----------------------------------------------------------------------------------------------------
# 네이버 증권에서 삼성전자, 삼성에스디에스 재무 정보 크롤링
#-----------------------------------------------------------------------------------------------------
SamSDS_list=[]

driver=webdriver.Chrome()
driver.get('https://finance.naver.com/item/coinfo.naver?code=018260')

driver.switch_to.frame('coinfo_cp')

html=driver.page_source
soup=BeautifulSoup(html,'html.parser')

result1=soup.find_all('tbody')
result2=result1[17].find_all('td',{'class':'num'})
print(result2)
