#---------------------------------------------------------------------------------
# it 대표주 - 삼성전자, 삼성에스디에스
# PER, PBR, ROE, 영업이익
# 2021~2023 추이 분석
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
SamSDS_list=[] # code=018260
SamEL_list=[] # code=005930

driver=webdriver.Chrome()
driver.get('https://finance.naver.com/item/coinfo.naver?code=005930')

driver.switch_to.frame('coinfo_cp')

html=driver.page_source
soup=BeautifulSoup(html,'html.parser')

body=soup.find_all('tbody')
result=body[17].find_all('td',{'class':'num'})

#-----------------------------------------------------------------------------------------------------
# 필요 데이터 추출
#-----------------------------------------------------------------------------------------------------
for _ in range(len(result)):
    span_string = str(result[_])
    remove_span = re.split(r'<td class="num">|</td>|<span class="cBk">|<span class="cUp">|</span>', span_string)
    for word in remove_span:
        if word !='':
            SamEL_list.append(word)
#print(SamEL_list)

#-----------------------------------------------------------------------------------------------------
# DataFrame 만들기
#-----------------------------------------------------------------------------------------------------
SEL_dict = {'OP':[SamEL_list[2],SamEL_list[12],SamEL_list[22]],
            'PER':[SamEL_list[5],SamEL_list[15],SamEL_list[25]],
            'PBR':[SamEL_list[6],SamEL_list[16],SamEL_list[26]],
            'ROE':[SamEL_list[7],SamEL_list[17],SamEL_list[27]]}

SEL_DF=pd.DataFrame(SEL_dict,index=['2021','2022','2023'])
print(SEL_DF)

#-----------------------------------------------------------------------------------------------------
# DF를 csv 파일로 저장
#-----------------------------------------------------------------------------------------------------
SEL_DF.to_csv("SEL_DF.csv", index = False)