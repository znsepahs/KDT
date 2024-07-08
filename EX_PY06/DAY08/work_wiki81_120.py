#81. scores의 좌측 8개 값을 가변변수 이용하여 valid_score에 저장
scores = [8.8, 8.9, 8.7, 9.2, 9.3, 9.7, 9.9, 9.5, 7.8, 9.4]
*valid_score,a,b=scores
#82. scores의 우측 8개 값을 가변변수 이용하여 valid_score에 저장
a,b,*valid_score=scores
#83. scores의 중앙 8개 값을 가변변수 이용하여 valid_score에 저장
a,*valid_score,b=scores
#84. temp라는 이름의 빈 딕셔너리 만들기
temp={}
#85. 자료를 이용하여 딕셔너리 만들기
name_price={'메로나':1000,'폴라포':1200,'빵빠레':1800}
#86. 딕셔너리에 키=값 추가하기
name_price['죠스바']=1200
name_price['월드콘']=1500
#87. 딕셔너리 키=값 출력하기 #실행 예:메로나 가격: 1000
print("메로나 가격: ", name_price['메로나'])
#88. 딕셔너리 값 수정하기 / 메로나가격 1300으로
name_price['메로나']=1300
#89. 딕셔너리 키 삭제하기 / 메로나 삭제
del name_price['메로나']
#90. 다음 코드에서 에러 발생 원인 찾기
# >> icecream = {'폴라포': 1200, '빵빠레': 1800, '월드콘': 1500, '메로나': 1000}
# >> icecream['누가바']
# Traceback (most recent call last):
#   File "<pyshell#69>", line 1, in <module>
#     icecream['누가바']
# KeyError: '누가바'
# 존재하지 않는 키를 인덱싱하여 오류 발생
#91. 아래의 표에서, 아이스크림 이름을 키값으로, (가격, 재고) 리스트를 딕셔너리의 값으로 저장하라. 딕셔너리의 이름은 inventory로 한다.
inventory = {"메로나": [300, 20], "비비빅": [400, 3], "죠스바": [250, 100]}
#92. 딕셔너리 인덱싱 / 위의 딕셔너리에서 메로나 가격 출력하기. 실행 예시:300 원
print(inventory["메로나"][0], '원')
#93. 메로나의 재고를 화면에 출력하라
print(inventory["메로나"][1], '개')
#94. inventory 딕셔너리에 아래 데이터를 추가하라 이름:월드콘 가격:500 재고:7
inventory["월드콘"]=[500, 7]
#95. 딕셔너리 keys() 메서드 / 아래 딕셔너리의 키로 구성된 리스트 만들기 
icecream = {'탱크보이': 1200, '폴라포': 1200, '빵빠레': 1800, '월드콘': 1500, '메로나': 1000}
name=list(icecream.keys())
print(name)
#96. 딕셔너리 values() 메서드 / 딕셔너리의 값으로만 구성된 리스트 만들기
price=list(icecream.values())
print(price)
#97. icecream 딕셔너리에서 아이스크림 판매 금액의 총합을 출력
print(sum(price))
#98. 딕셔너리 update() 메서드
new_product = {'팥빙수':2700, '아맛나':1000}
icecream.update(new_product)
#99. zip과 dict /  두 개의 튜플을 하나의 딕셔너리로 변환. keys를 키로, vals를 값으로, 딕셔너리 이름  result
keys=("apple", "pear", "peach")
vals=(300, 250, 400)
result=dict(zip(keys, vals))
#100. date와 close_price 두 개의 리스트를 close_table 이름의 딕셔너리로 생성하라.
date = ['09/05', '09/06', '09/07', '09/08', '09/09']
close_price = [10500, 10300, 10100, 10800, 11000]
close_table=dict(zip(date,close_price))
#101. 파이썬에서 True 혹은 False를 갖는 데이터 타입? boolean, bool
#102. 코드의 출력 결과를 예상하라 print(3 == 5)
#정답: False
#103. 코드의 출력 결과를 예상하라 print(3 < 5)
#정답: True
#104. 코드의 결과를 예상하라. x = 4, print(1 < x < 5)
#정답: True
#105. 코드의 결과를 예상하라. print ((3 == 3) and (4 != 3))
#정답: True
#106. 코드에서 에러가 발생하는 원인에 대해 설명하라. print(3 => 4)
#정답: >=로 변경하면 작동
#107. 아래 코드의 출력 결과를 예상하라
if 4 < 3:
    print("Hello World")
#정답: 조건문이 거짓이라 아무것도 출력되지 않음
#108. 아래 코드의 출력 결과를 예상하라
if 4 < 3:
    print("Hello World.")
else:
    print("Hi, there.")
#정답: if 조건문은 거짓이라 실행안되고 else의 print함수가 실행
#109. 아래 코드의 출력 결과를 예상하라
if True :
    print ("1")
    print ("2")
else :
    print("3")
print("4")
#정답: if 조건문이 참이라 들여쓰기 된 코드가 차례대로 실행되어 "1","2"출력, else조건 실행안됨, 마지막의 print함수 실행되어 "4"출력
#110. 아래 코드의 출력 결과 예상
if True :
    if False:
        print("1")
        print("2")
    else:
        print("3")
else :
    print("4")
print("5")
#정답: 첫번째 if문 참 => 이중조건문 if는 false이므로 실행안되고 이중조건문 else: print("3") 실행,조건문 마지막의 print("5") 실행
#111. 사용자로부터 입력받은 문자열을 두 번 출력
txt=input()
print(txt*2)
#112. 사용자로부터 하나의 숫자를 입력받고, 입력 받은 숫자에 10을 더해 출력
num1=int(input())
print(num1+10)
#113. 사용자로부터 하나의 숫자를 입력 받고 짝수/홀수를 판별하라.
num2=int(input())
if num2%2: print("홀수입니다")
else : print("짝수입니다")
#114. 사용자로부터 값을 입력받은 후 20을 더한 값을 출력 / 사용자가 입력한 값과 20을 더한 계산 값이 255를 초과하는 경우 255를 출력
num3=int(input())
if (num3+20)>255: print(255)
else : print(num3+20)
#115. 사용자로부터 값을 입력받은 후 20을 뺸 값을 출력 / 결괏값의 범위는 0~255 / 0보다 작을 경우 0을 출력하고 255보다 큰 값이 되는 경우 255를 출력
num4=int(input())
if num4-20<0:print(0)
elif num4-20>255:print(255)
else: print(num4-20)
#116. 사용자로부터 입력 받은 시간이 정각인지 판별하라.
#예: 현재시간:02:00 / 정각 입니다. 현재시간:03:10 / 정각이 아닙니다
time=input().split(':')
if time[-1]=='00': print("정각 입니다.")
else : print("정각이 아닙니다.")
#117. 사용자로 입력받은 단어가 fruit 리스트에 포함되어 있는지를 확인하고 포함되었다면 "정답입니다"를 아닐 경우 "오답입니다" 출력
fruit = ["사과", "포도", "홍시"]
check=input()
if check in fruit: print("정답입니다")
else: print("오답입니다.")
#118. 투자 경고 종목 리스트가 있다. 사용자로부터 종목명을 입력 받은 후 해당 종목이 투자 경고 종목이라면 '투자 경고 종목입니다'를 
# 아니면 "투자 경고 종목이 아닙니다."를 출력하는 프로그램을 작성하라.
warn_investment_list = ["Microsoft", "Google", "Naver", "Kakao", "SAMSUNG", "LG"]
inv=input()
if inv in warn_investment_list: print("투자 경고 종목입니다.")
else : print("투자 경고 종목이 아닙니다.")
#119. fruit 딕셔너리가 있다. 사용자가 입력한 값이 딕셔너리 키 (key) 값에 포함되었다면 "정답입니다"를 아닐 경우 "오답입니다" 출력하라.
fruit = {"봄" : "딸기", "여름" : "토마토", "가을" : "사과"}
kcheck=input()
if kcheck in fruit.keys(): print("정답입니다.")
else : print("오답입니다.")
#120. 사용자가 입력한 값이 딕셔너리 값 (value)에 포함되었다면 "정답입니다"를 아닐 경우 "오답입니다" 출력하라.
vcheck=input()
if vcheck in fruit.values(): print("정답입니다.")
else : print("오답입니다.")