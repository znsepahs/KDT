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