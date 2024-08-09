# 웹 페이지 가져오기
from urllib.request import urlopen

html=urlopen('https://daangn.com/hot_articles')
print(type(html))
print(html.read()) # 웹 사이트의 데이터 (byte 형태)
