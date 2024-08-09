import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np

conn = pymysql.connect(host='172.20.139.58', user='member3', password='1234',db='sqlteam4_db', charset = 'utf8')

query1='''select area.city,
y2015.buscardfee as '2015',
y2016.buscardfee as '2016',
y2017.buscardfee as '2017',
y2018.buscardfee as '2018',
y2019.buscardfee as '2019',
y2020.buscardfee as '2020',
y2021.buscardfee as '2021',
y2022.buscardfee as '2022',
y2023.buscardfee as '2023'
from area
inner join y2015 on y2015.city = area.city
inner join y2016 on y2016.city = area.city
inner join y2017 on y2017.city = area.city
inner join y2018 on y2018.city = area.city
inner join y2019 on y2019.city = area.city
inner join y2020 on y2020.city = area.city
inner join y2021 on y2021.city = area.city
inner join y2022 on y2022.city = area.city
inner join y2023 on y2023.city = area.city;
'''
query2='''select
cpi.`2015`, cpi.`2016`, cpi.`2017`, cpi.`2018`, cpi.`2019`, cpi.`2020`,cpi.`2021`,cpi.`2022`,cpi.`2023`
from area as a
inner join cpi on cpi.city = a.city;'''

query3='''select area.city,
y2015.buscashfee as '2015',
y2016.buscashfee as '2016',
y2017.buscashfee as '2017',
y2018.buscashfee as '2018',
y2019.buscashfee as '2019',
y2020.buscashfee as '2020',
y2021.buscashfee as '2021',
y2022.buscashfee as '2022',
y2023.buscashfee as '2023'
from area
inner join y2015 on y2015.city = area.city
inner join y2016 on y2016.city = area.city
inner join y2017 on y2017.city = area.city
inner join y2018 on y2018.city = area.city
inner join y2019 on y2019.city = area.city
inner join y2020 on y2020.city = area.city
inner join y2021 on y2021.city = area.city
inner join y2022 on y2022.city = area.city
inner join y2023 on y2023.city = area.city;
'''

cur1 = conn.cursor(pymysql.cursors.DictCursor)
cur1.execute(query1)
rows=cur1.fetchall()

buscard_df=pd.DataFrame(rows)
#print(buscard_df)

period=range(2015,2024)
seoul=[buscard_df.iloc[0,_] for _ in range(1,len(buscard_df.columns))] 
gwang=[buscard_df.iloc[1,_] for _ in range(1,len(buscard_df.columns))]
daegu=[buscard_df.iloc[2,_] for _ in range(1,len(buscard_df.columns))]
sungsim=[buscard_df.iloc[3,_] for _ in range(1,len(buscard_df.columns))]
busan=[buscard_df.iloc[4,_] for _ in range(1,len(buscard_df.columns))]
incheon=[buscard_df.iloc[6,_] for _ in range(1,len(buscard_df.columns))]

ax1=plt.subplot(2, 3, 1)                
plt.plot(period,seoul,'bo-',label='버스요금(카드)')
plt.title("[서울 버스 요금]")
plt.xlabel("YEAR")
plt.ylabel("금액")
plt.xticks(visible=False)
plt.legend()

ax2=plt.subplot(2, 3, 2, sharex=ax1)               
plt.plot(period,gwang,'ro-',label='버스요금(카드)')
plt.title("[광주 버스 요금]")
plt.xlabel("YEAR")
plt.ylabel("금액")
plt.xticks(visible=False)
plt.legend()

ax3=plt.subplot(2, 3, 3, sharex=ax1)               
plt.plot(period,daegu,'yo-',label='버스요금(카드)')
plt.title("[대구 버스 요금]")
plt.xlabel("YEAR")
plt.ylabel("금액")
plt.xticks(visible=False)
plt.legend()

ax4=plt.subplot(2, 3, 4, sharex=ax1)               
plt.plot(period,sungsim,'go-',label='버스요금(카드)')
plt.title("[대전 버스 요금]")
plt.xlabel("YEAR")
plt.ylabel("금액")
plt.legend()

ax5=plt.subplot(2, 3, 5, sharex=ax1)               
plt.plot(period,busan,'ko-',label='버스요금(카드)')
plt.title("[부산 버스 요금]")
plt.xlabel("YEAR")
plt.ylabel("금액")
plt.legend()

ax6=plt.subplot(2, 3, 6, sharex=ax1)               
plt.plot(period,incheon,'mo-',label='버스요금(카드)')
plt.title("[인천 버스 요금]")
plt.xlabel("YEAR")
plt.ylabel("금액")
plt.legend()

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------------
cur2 = conn.cursor(pymysql.cursors.DictCursor)
cur2.execute(query2)
rows=cur2.fetchall()

Cpi_df=pd.DataFrame(rows)
print(Cpi_df)
period=range(2015,2024)
S_cpi=[Cpi_df.iloc[0,_] for _ in range(len(Cpi_df.columns))]
G_cpi=[Cpi_df.iloc[1,_] for _ in range(len(Cpi_df.columns))]
D_cpi=[Cpi_df.iloc[2,_] for _ in range(len(Cpi_df.columns))]
SS_cpi=[Cpi_df.iloc[3,_] for _ in range(len(Cpi_df.columns))]
B_cpi=[Cpi_df.iloc[4,_] for _ in range(len(Cpi_df.columns))]
I_cpi=[Cpi_df.iloc[6,_] for _ in range(len(Cpi_df.columns))]

period=range(2015,2024)
seoul=[buscard_df.iloc[0,_] for _ in range(1,len(buscard_df.columns))] 
gwang=[buscard_df.iloc[1,_] for _ in range(1,len(buscard_df.columns))]
daegu=[buscard_df.iloc[2,_] for _ in range(1,len(buscard_df.columns))]
sungsim=[buscard_df.iloc[3,_] for _ in range(1,len(buscard_df.columns))]
busan=[buscard_df.iloc[4,_] for _ in range(1,len(buscard_df.columns))]
incheon=[buscard_df.iloc[6,_] for _ in range(1,len(buscard_df.columns))]

bx1=plt.subplot(2, 3, 1)                
plt.plot(period,S_cpi,'bo-',label='CPI')
plt.title("[서울 CPI]")
plt.xlabel("YEAR")
plt.ylabel("소비자 물가 지수")
plt.xticks(visible=False)
plt.legend()

bx2=plt.subplot(2, 3, 2, sharex=bx1)               
plt.plot(period,G_cpi,'ro-',label='CPI')
plt.title("[광주 CPI]")
plt.xlabel("YEAR")
plt.ylabel("소비자 물가 지수")
plt.xticks(visible=False)
plt.legend()

plt.subplot(2, 3, 3, sharex=ax1)               
plt.plot(period,D_cpi,'yo-',label='CPI')
plt.title("[대구 CPI]")
plt.xlabel("YEAR")
plt.ylabel("소비자 물가 지수")
plt.xticks(visible=False)
plt.legend()

plt.subplot(2, 3, 4, sharex=ax1)               
plt.plot(period,SS_cpi,'go-',label='CPI')
plt.title("[대전 CPI]")
plt.xlabel("YEAR")
plt.ylabel("소비자 물가 지수")
plt.legend()

plt.subplot(2, 3, 5, sharex=ax1)               
plt.plot(period,B_cpi,'ko-',label='CPI')
plt.title("[부산 CPI]")
plt.xlabel("YEAR")
plt.ylabel("소비자 물가 지수")
plt.legend()

plt.subplot(2, 3, 6, sharex=ax1)               
plt.plot(period,I_cpi,'mo-',label='CPI')
plt.title("[인천 CPI]")
plt.xlabel("YEAR")
plt.ylabel("소비자 물가 지수")
plt.legend()

plt.tight_layout()
plt.show()
#-----------------------------------------------------------------------------------
cur3 = conn.cursor(pymysql.cursors.DictCursor)
cur3.execute(query3)
rows=cur3.fetchall()

buscash_df=pd.DataFrame(rows)
#print(buscash_df)

period=range(2015,2024)
seoul1=[buscash_df.iloc[0,_] for _ in range(1,len(buscash_df.columns))] 
gwang1=[buscash_df.iloc[1,_] for _ in range(1,len(buscash_df.columns))]
daegu1=[buscash_df.iloc[2,_] for _ in range(1,len(buscash_df.columns))]
sungsim1=[buscash_df.iloc[3,_] for _ in range(1,len(buscash_df.columns))]
busan1=[buscash_df.iloc[4,_] for _ in range(1,len(buscash_df.columns))]
incheon1=[buscash_df.iloc[6,_] for _ in range(1,len(buscash_df.columns))]

ax1=plt.subplot(2, 3, 1)                
plt.plot(period,seoul1,'bo-',label='버스요금(현금)')
plt.title("[서울 버스 요금]")
plt.xlabel("YEAR")
plt.ylabel("금액")
plt.xticks(visible=False)
plt.legend()

ax2=plt.subplot(2, 3, 2, sharex=ax1)               
plt.plot(period,gwang1,'ro-',label='버스요금(현금)')
plt.title("[광주 버스 요금]")
plt.xlabel("YEAR")
plt.ylabel("금액")
plt.xticks(visible=False)
plt.legend()

plt.subplot(2, 3, 3, sharex=ax1)               
plt.plot(period,daegu1,'yo-',label='버스요금(현금)')
plt.title("[대구 버스 요금]")
plt.xlabel("YEAR")
plt.ylabel("금액")
plt.xticks(visible=False)
plt.legend()

plt.subplot(2, 3, 4, sharex=ax1)               
plt.plot(period,sungsim1,'go-',label='버스요금(현금)')
plt.title("[대전 버스 요금]")
plt.xlabel("YEAR")
plt.ylabel("금액")
plt.legend()

plt.subplot(2, 3, 5, sharex=ax1)               
plt.plot(period,busan1,'ko-',label='버스요금(현금)')
plt.title("[부산 버스 요금]")
plt.xlabel("YEAR")
plt.ylabel("금액")
plt.legend()

plt.subplot(2, 3, 6, sharex=ax1)               
plt.plot(period,incheon1,'mo-',label='버스요금(현금)')
plt.title("[인천 버스 요금]")
plt.xlabel("YEAR")
plt.ylabel("금액")
plt.legend()

plt.tight_layout()
plt.show()
#-----------------------------------------------------------------------------------
# 발표 개요
#-----------------------------------------------------------------------------------
# 데이터
#-----------------------------------------------------------------------------------
# 2015~2024 6대 권역 버스요금
# 2015~2024 6대 권역 CPI
#-----------------------------------------------------------------------------------
# 결론
# - 물가가 오른다고 시내버스 요금이 비례해서 오르지는 않는다.
# - 운임요금만으로는 적자가 발생, 지자체의 지원금과 버스 준공영제를 통해 사업 유지
# - 공공 인프라의 가치를 되새기고, 오르는 물가때문에 지갑 사정이 좋지 않다면 자가용 운용을 지양하고
#    시내버스 이용이 용이한 6대 권역에서 시내버스로 출퇴근 하는 것이 좋다고 판단.
#-----------------------------------------------------------------------------------