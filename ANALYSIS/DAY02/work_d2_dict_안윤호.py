#----------------------------------------------------------------------------
# 메뉴 출력용 사용자 함수 만들기
# 브레이크 조건이 6을 입력받는 while 반복문
# 필요한 데이터를 변수로 저장
# 요구하는 데이터를 프린트하기
#----------------------------------------------------------------------------

def print_menu():
    print('-----------------------------------------')
    print('1. 전체 데이터 출력')
    print('2. 수도 이름 오름차순 출력')
    print('3. 모든 도시의 인구수 내림차순 출력')
    print('4. 특정 도시의 정보 출력')
    print('5. 대륙별 인구수 계산 및 출력')
    print('6. 프로그램 종료')
    print('-----------------------------------------')

data={'수도이름':['Seoul','Tokyo','Beijing','London','Berlin','Mexico City'],
    '국가명':['South Korea','Japan','China','United Kingdom','Germany','Mexico'],
    '대륙':['Asia','Asia','Asia','Europe','Europe','America'],
    '인구수':['9655000','14110000','21540000','14800000','3426000','21200000']}

data1={'Seoul':['South Korea','Asia','9655000'],
       'Tokyo':['Japan','Asia','14110000'],
       'Beijing':['China','Asia','21540000'],
       'London':['United Kingdom','Europe','14800000'],
       'Berlin':['Germany','Europe','3426000'],
       'Mexico City':['Mexico','America','21200000']}

data2={'Seoul':9655000,
       'Tokyo':14110000,
       'Beijing':21540000,
       'London':14800000,
       'Berlin':3426000,
       'Mexico City':21200000}

need1=list(data1.keys())
need1.sort()

need2=sorted(data2.items(),key=lambda x:x[1],reverse=True)

while True:
    print_menu()
    info=input("메뉴를 입력하세요.")

    a=0
    b=0
    c=0
    if info=='1':
        for _ in range(6):
            _+=a
            print('[',_+1,']',data['수도이름'][_],':','[',data['국가명'][_],' ,',data['대륙'][_],' ,',data['인구수'][_],']',sep='')

    elif info=='2':
        for _ in range(6):
            _+=b
            print('[',_+1,']',need1[_],':',data1[need1[_]],sep='')

    elif info=='3':
        for _ in range(6):
            _+=c
            print('[',_+1,']',need2[_][0],':',need2[_][1],sep='')

    elif info=='4':
        city_name=input('출력할 도시 이름을 입력하세요: ')
        if city_name in data['수도이름']:
            print(f'도시:{city_name} \n국가:{data1[city_name][0]} 대륙:{data1[city_name][1]} 인구수:{data1[city_name][2]}')

        else: print(f'도시이름: {city_name}은 key에 없습니다.')

    elif info=='5':
        continent=input('대륙 이름을 입력하세요(Asia, Europe, America): ')
        if continent == 'Asia':
            print(f"Seoul: {data1['Seoul'][2]}\nTokyo: {data1['Tokyo'][2]}\nBeijing: {data1['Beijing'][2]}")
            print(f"Asia 전체 인구수: {int(data1['Seoul'][2])+int(data1['Tokyo'][2])+int(data1['Beijing'][2])}")
        elif continent == 'Europe':
            print(f"London: {data1['London'][2]}\nBerlin: {data1['Berlin'][2]}")
            print(f"Europe 전체 인구수: {int(data1['London'][2])+int(data1['Berlin'][2])}")
        elif continent == 'America':
            print(f"Mexico City: {data1['Mexico City'][2]}")
            print(f"America 전체 인구수: {int(data1['Mexico City'][2])}")

    elif info=='6':
        print("프로그램을 종료합니다.")
        break