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
import collections

if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

#-----------------------------------------------------------------------------------------------------
# 사람인에서 500개 기업들의 구인 태그 정보 크롤링
#-----------------------------------------------------------------------------------------------------
tag_list=[]
try:
    for a in range(11,16):
        search_num=a
        url = f'https://www.saramin.co.kr/zf_user/jobs/list/job-category?page={search_num}&cat_kewd=2248%2C82%2C83%2C109%2C116%2C107%2C106&search_optional_item=n&search_done=y&panel_count=y&preview=y&isAjaxRequest=0&page_count=100&sort=RL&type=job-category&is_param=1&isSearchResultEmpty=1&isSectionHome=0&searchParamCount=1'

        urlrequest = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(urlrequest)

        soup = BeautifulSoup(html, 'html.parser')
        contents = soup.find_all('span', {'class':'job_sector'})
        for _ in range(100):
            span_string = str(contents[_])
            remove_span = re.split(r'<span>|</span>', span_string)
            for word in remove_span:
                if word !='' and word != '<span class="job_sector">\n' and word != ' 외                    ':
                    tag_list.append(word)
except Exception as e:
    print(e)

#-----------------------------------------------------------------------------------------------------
# 리스트를 txt 파일로 저장
#-----------------------------------------------------------------------------------------------------
with open("tag_list.txt", "w", encoding="UTF-8") as file:
    for item in tag_list:
        file.write(item + "\n")

#-----------------------------------------------------------------------------------------------------
# 리스트를 csv 파일로 저장
#-----------------------------------------------------------------------------------------------------
df=pd.DataFrame(tag_list, columns=['tag'])
df.to_csv("tag_list.csv", index = False)

#-----------------------------------------------------------------------------------------------------
# 시각화 - wordcloud
#-----------------------------------------------------------------------------------------------------
if platform.system() =='Windows':
    path=r'c:\Windows\Fonts\malgun.ttf'

text=open('tag_list.txt',encoding="UTF-8").read()
img_mask=np.array(Image.open('cloud.png'))
wc=WordCloud(font_path=path, width=400, height=400, background_color="white", max_font_size=200,
             repeat=True, colormap='inferno', mask=img_mask).generate(text)

plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(wc)
plt.show()

#-----------------------------------------------------------------------------------------------------
# 발표 개요
#-----------------------------------------------------------------------------------------------------
# 1. 사람인에서 IT 관련 분야 500개 기업 구인 공고를 크롤링 / 모집 태그를 wordcloud로 시각화 하여 대략적인 추세 파악

# 2. IT 분야 기업들의 최근 주요재무지표를 분석하여 구직 추천 / 비추천

# 2-1. 주요 용어 간략 설명
# 2-1-1. PER : 현재 주가(Price) / 주당순이익(EPS). 주가이익비율
#               PER이 낮으면 주당 순이익이 크다는 의미 => 저평가.

# 2-1-2. PBR : 현재 주가(Price) / 주당장부가치(BPS). 주가순자산비율
#               1보다 높으면 고평가, 낮으면 저평가. 장부가치에 비해 실제 시장가격의 높낮음을 반영
#               
# 2-1-3. ROE : 순이익 / 자기자본 * 100. 자기자본이익률
#               경영자가 주주의 자본으로 얼마나 효율적으로 이익을 내고 있는가 

# 3. IT 대표주 - 삼성전자, 삼성에스디에스

# 3-1. 2021~2023 동안 영업이익, PER, PBR, ROE 추이 분석
# 3-2. 삼성전자가 2023년 영업이익이 주춤한 모습을 보이지만 그럼에도 가장 역량있는 기업이다.
