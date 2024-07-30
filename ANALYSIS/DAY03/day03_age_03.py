import csv
import matplotlib.pyplot as plt
import re
import koreanize_matplotlib

f = open('age.csv', encoding='euc_kr') 
data = csv.reader(f)
result=[]
city=''
# row[0]: 행정구역
for row	in data:
    if '산격3동' in row[0]:
        str_list=re.split('[()]',row[0]) # '산격3동'이 포함된 자료만 출력
        city=str_list[0]
        for data in row[3:]:
            data=data.replace(',','')
            result.append(int(data))
        
f.close()
print(result)

plt.title(f'{city} 인구현황')
plt.xlabel('나이')
plt.ylabel('인구수')
plt.plot(result)
plt.show()