from bs4 import BeautifulSoup
from urllib.request import urlopen
import collections

collections.Callable=collections.abc.Callable

base = 'https://finance.naver.com'
market_link = '/sise/sise_market_sum.naver'
stock_url = base + market_link
html = urlopen(stock_url)
soup = BeautifulSoup(html, 'html.parser')

com_name=[]
com_link=[]

s_table=soup.find('table',{'class':'type_2'})
s_body=s_table.find('tbody')
td_list = s_body.find_all('td')

for i in range(len(td_list)):
        if td_list[i].select_one('a.tltle'):
                name = td_list[i].text
                link = td_list[i].find('a',{'class':'tltle'}).attrs['href']
                com_name.append(name)
                com_link.append(link)
                if len(com_link) == 10:break
# print(com_name)
# print(com_link)

name_list = []
code_list = []
pre_price_list = []
last_price_list = []
price_list = []
max_list = []
min_list = []

for i in range(10):
        url = base + com_link[i]
        html = urlopen(url)
        soup = BeautifulSoup(html, 'html.parser')

        data = soup.find('dl', {'class':'blind'}).text.split()

        name_list.append(data[11])
        code_list.append(data[13])
        pre_price_list.append(data[16])
        last_price_list.append(data[24])
        price_list.append(data[26])
        max_list.append(data[28])
        min_list.append(data[30])

# 주가를 보여주는 함수
def show_price(key):
        key = int(key)-1
        print(com_link[key])
        print(f"종목명: {name_list[key]}")
        print(f"종목코드: {code_list[key]}")
        print(f"현재가: {pre_price_list[key]}")
        print(f"전일가: {last_price_list[key]}")
        print(f"시가: {price_list[key]}")
        print(f"고가: {max_list[key]}")
        print(f"저가: {min_list[key]}")
        

while True:
        print('-'*25)
        print('[ 네이버 코스피 상위 10개 기업 목록 ]')
        print('-'*25)
        for i in range(10):
                print(f"[{i+1:2}] {com_name[i]}")
        
        key = input('주가를 검색할 기업의 번호를 입력하세요(-1: 종료): ')

        if key == '-1':
                print('프로그램 종료')
                break
        elif int(key) in [x for x in range(1, 11)]:
                show_price(key)