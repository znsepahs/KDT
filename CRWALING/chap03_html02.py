from urllib.request import urlopen
from bs4 import BeautifulSoup

html=urlopen('http://www.pythonscraping.com/pages/warandpeace.html')
soup=BeautifulSoup(html, 'html.parser')

# 등장인물의 이름: 녹색
name_list=soup.find_all('span',{'class':'green'})
for name in name_list:
    print(name.string)

prince_list=soup.find_all(string='the prince')
print(prince_list)
print('the prince count:', len(prince_list))

