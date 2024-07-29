#------------------------------------------------------------------------------------------------------------------
# 공공데이터 1일차 과제
#------------------------------------------------------------------------------------------------------------------
# 1. 대구 기온 데이터에서 시작 연도, 마지막 연도를 입력하고 특정 월의 최고 기온 및 최저 기온의 평균값을 구하고 그래프로 표현
# - daegu-utf8.csv 또는 daegu-utf8-df.csv 파일 이용
# - 데이터 구조 : ['날짜', '지점', '평균기온', '최저기온', '최고기온’]
# - 화면에서 측정할 달을 입력 받아서 진행
# - 해당 기간 동안 최고기온 평균값 및 최저기온 평균값 계산
# - (1) 최고기온 및 최저기온 데이터를 이용하여 입력된 달의 각각 평균값을 구함 
# - (2) 문자열 형태의 ‘날짜’ 열의 데이터는 datetime 으로 변경함:
# - 하나의 그래프 안에 2개의 꺾은선 그래프로 결과를 출력 
# - (1) 마이너스 기호 출력 깨짐 방지
# - (2) 입력된 월을 이용하여 그래프의 타이틀 내용 변경
# - (3) 최고 온도는 빨간색, 최저 온도는 파란색으로 표시하고 각각 마커 및 legend를 표시
#------------------------------------------------------------------------------------------------------------------
# 실행 결과
## 시작 연도를 입력하세요: 2014
## 마지막 연도를 입력하세요: 2023
## 기온 변화를 측정할 달을 입력하세요: 12 
## 2014 년부터 2023 년까지 12 월의 기온 변화 
## 12 월 최저기온 평균: -2.8, 0.5, -0.6, -4.0, -2.3, -1.0, -3.1, -2.1, -4.0, -1.3 
## 12 월 최고기온 평균: 5.8, 9.0, 8.7, 5.8, 7.0, 9.0, 7.2, 8.9, 5.4, 8.6
#------------------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

def draw_two_plots(title, x_data, max_temp_mean, label_y1, min_temp_mean, label_y2):

    plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(10,4))
    plt.plot(x_data,max_temp_mean, marker='o', markersize=6, color='r', label=label_y1)
    plt.plot(x_data,min_temp_mean, marker='s', markersize=6, color='b', label=label_y2)

    plt.title(title)
    plt.legend()
    plt.show()

def main():
    search_year1=int(input('시작 연도를 입력하세요: '))
    search_year2=int(input('마지막 연도를 입력하세요: '))
    search_month=int(input('기온 변화를 측정할 달을 입력하세요: '))

    weather_df=pd.read_csv('daegu-utf8-df.csv',encoding='utf-8-sig')
    weather_df['날짜']=pd.to_datetime(weather_df['날짜'],format='%Y-%m-%d')

    search_month_max_temp_list=[0]*10
    search_month_min_temp_list=[0]*10

    first_decade=search_year1
    second_decade=search_year2

    print(f'{first_decade} 년부터 {second_decade} 년까지 {search_month} 월의 기온 변화')

    for year in range(10):
        max_temp_df=weather_df[(weather_df['날짜'].dt.year==first_decade + year) &
                                   (weather_df['날짜'].dt.month==search_month)]
        search_month_max_temp_list[year]=round(max_temp_df['최고기온'].mean(),1)

        min_temp_df=weather_df[(weather_df['날짜'].dt.year==first_decade + year) &
                                   (weather_df['날짜'].dt.month==search_month)]
        search_month_min_temp_list[year]=round(min_temp_df['최저기온'].mean(),1)
    
    print(f'{search_month} 월 최저기온 평균 : {search_month_min_temp_list}')
    print(f'{search_month} 월 최고기온 평균: {search_month_max_temp_list}')

    x_data=[i for i in range(10)]
    draw_two_plots(f'{search_month}월 최고 기온 비교', x_data,
                   search_month_max_temp_list, str(search_month)+'월 최고온도',
                   search_month_min_temp_list, str(search_month)+'월 최저온도')

main()