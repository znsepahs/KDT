from urllib.request import urlopen
from bs4 import BeautifulSoup

html=urlopen('http://www.pythonscraping.com/pages/page1.html')
bs=BeautifulSoup(html.read(), 'html.parser') # 클래스의 생성자 : 객체 생성
# print(bs) # bs 전체 출력
# print(bs.h1) # bs 중에서 h1 태그 찾기 / 쥬피터 마크다운에서 썼던 "### 문자열" 이런거
# print(bs.h1.string) # bs 중에서 h1 태그 찾고 h1 태그 내부의 string 출력
print(bs.div) # <div> 텍스트 .... </div>
print(bs.div.text)
