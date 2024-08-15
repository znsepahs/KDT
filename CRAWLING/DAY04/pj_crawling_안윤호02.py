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
import koreanize_matplotlib
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

SDS_DF=pd.read_csv('SDS_DF.csv')
SEL_DF=pd.read_csv('SEL_DF.csv')
#-----------------------------------------------------------------------------------------------------
# DF를 csv 파일로 저장
#-----------------------------------------------------------------------------------------------------
# SEL_DF.to_csv("SEL_DF.csv", index = False)

#-----------------------------------------------------------------------------------------------------
# 시각화 - 선 그래프 subplot
#-----------------------------------------------------------------------------------------------------
# 삼성에스디에스
period=range(2021,2024)
SDS_OP=[SDS_DF.iloc[_,0] for _ in range(3)]
SDS_PER=[SDS_DF.iloc[_,1] for _ in range(3)]
SDS_PBR=[SDS_DF.iloc[_,2] for _ in range(3)]
SDS_ROE=[SDS_DF.iloc[_,3] for _ in range(3)]

ax1=plt.subplot(2, 2, 1)                
plt.plot(period,SDS_OP,'bo-',label='영업이익')
plt.title("[삼성에스디에스 영업이익 2021~2023]")
plt.xlabel("YEAR")
plt.ylabel("억원, %")
plt.xticks(visible=False)
plt.legend()

ax2=plt.subplot(2, 2, 2, sharex=ax1)               
plt.plot(period,SDS_PER,'ro-',label='PER')
plt.title("[삼성에스디에스 PER 2021~2023]")
plt.xlabel("YEAR")
plt.ylabel("배")
plt.xticks(visible=False)
plt.legend()

ax3=plt.subplot(2, 2, 3, sharex=ax1)               
plt.plot(period,SDS_PBR,'yo-',label='PBR')
plt.title("[삼성에스디에스 PBR 2021~2023]")
plt.xlabel("YEAR")
plt.ylabel("배")
plt.xticks(visible=False)
plt.legend()

ax4=plt.subplot(2, 2, 4, sharex=ax1)               
plt.plot(period,SDS_ROE,'go-',label='ROE')
plt.title("[삼성에스디에스 ROE 2021~2023]")
plt.xlabel("YEAR")
plt.ylabel("%")
plt.legend()

plt.tight_layout()
plt.show()

# 삼성전자
period=range(2021,2024)
SEL_OP=[SEL_DF.iloc[_,0] for _ in range(3)]
SEL_PER=[SEL_DF.iloc[_,1] for _ in range(3)]
SEL_PBR=[SEL_DF.iloc[_,2] for _ in range(3)]
SEL_ROE=[SEL_DF.iloc[_,3] for _ in range(3)]

ax1=plt.subplot(2, 2, 1)                
plt.plot(period,SDS_OP,'bo-',label='영업이익')
plt.title("[삼성전자 영업이익 2021~2023]")
plt.xlabel("YEAR")
plt.ylabel("억원, %")
plt.xticks(visible=False)
plt.legend()

ax2=plt.subplot(2, 2, 2, sharex=ax1)               
plt.plot(period,SEL_PER,'ro-',label='PER')
plt.title("[삼성전자 PER 2021~2023]")
plt.xlabel("YEAR")
plt.ylabel("배")
plt.xticks(visible=False)
plt.legend()

ax3=plt.subplot(2, 2, 3, sharex=ax1)               
plt.plot(period,SEL_PBR,'yo-',label='PBR')
plt.title("[삼성전자 PBR 2021~2023]")
plt.xlabel("YEAR")
plt.ylabel("배")
plt.xticks(visible=False)
plt.legend()

ax4=plt.subplot(2, 2, 4, sharex=ax1)               
plt.plot(period,SEL_ROE,'go-',label='ROE')
plt.title("[삼성전자 ROE 2021~2023]")
plt.xlabel("YEAR")
plt.ylabel("%")
plt.legend()

plt.tight_layout()
plt.show()