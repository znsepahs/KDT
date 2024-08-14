from urllib.request import Request
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from urllib.parse import quote
from urllib.request import urlopen
from bs4 import BeautifulSoup
import collections

if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

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

print(tag_list)