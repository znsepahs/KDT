from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

query='ChatGPT'
query1=quote('챗지피티') # 한글 검색어 전달
url=f'https://search.naver.com/search.naver?where=view&sm=tab_jum&query={query1}'

html=urlopen(url)
soup=BeautifulSoup(html.read(),'html.parser')
blog_results=soup.select('a,title_link') # 검색 결과 타이틀
print('검색 결과수: ', len(blog_results))
search_count=len(blog_results)
desc_results=soup.select('a.dsc_link') # 검색 결과의 간단한 설명

for i in range(search_count):
    title=blog_results[i].text
    link=blog_results[i]['href']
    print(f'{title},[{link}]')
    print(desc_results[i].text)
    print('-'*80)