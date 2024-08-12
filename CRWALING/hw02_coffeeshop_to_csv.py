# 매장 찾기에서 1~50페이지까지 모든 매장의 정보를 스크레이핑
# • 지역, 매장명, 매장 주소, 전화번호
# • 수집된 정보는 csv 파일로 저장함
# • 결과물
# – csv파일: hollys_branches.csv (utf-8로 인코딩)

from urllib.request import urlopen
from bs4 import BeautifulSoup

html=urlopen('https://www.hollys.co.kr/store/korea/korStore2.do?pageNo=1&sido=&gugun=&store=')
soup=BeautifulSoup(html, 'html.parser')
result=soup.find_all('tbody')
