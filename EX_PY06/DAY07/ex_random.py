#------------------------------------------------------------------------
# 모듈 : 변수, 함수, 클래스가 들어있는 파이썬 파일
# 패키지 : 동일한 목적의 모듈들을 모은 것
#         여러개의 모듈 파일들 존재
# 모듈 사용법 : import 모듈파일명 <- 확장자 제외!
#------------------------------------------------------------------------
import random as rad

rad.random()

# 임의의 숫자를 생성 추출 하기
# 임의의 숫자 10개 생성
# => random() : 0.0<=~<1.0
for cnt in range(10):
    print(int(rad.random()*10))

# => randint(a,b) : a<=~<=b
for cnt in range(10):
    print(rad.randint(0,1))
print()
#------------------------------------------------------------------------
# [실습] 로또 프로그램을 만들어주세요.
# - 1~45 범위에서 중복되지 않는 6개 추출
#------------------------------------------------------------------------
# 반복 횟수 알 수 없음 / 종료조건 : 중복없는 숫자 6개
# list, dict, set ?

# list
lotto=[0,0,0,0,0,0]
idx=0
while True:
    num=rad.randint(1,45)
    if num not in lotto:
        lotto[idx]=num
        idx=idx+1
    if idx==6:break
print(lotto)
#dict
lotto={}
key=1
while True:
    num=rad.randint(1,45)
    if num not in lotto.values():
        lotto[key]=num
        key=key+1
    if key==7:break
print(lotto.values())
#set
lotto=set()
key=1
while len(lotto)<6:
    num=rad.randint(1,45)
    num_set=set([num])
    lotto.union(num_set)
    lotto=lotto.union(num_set)
print(lotto)

#set 타입의 add()메서드 : 원소추가, 중복 시 추가x
lotto=set()
while len(lotto)<6:
    num=rad.randint(1,45)
    lotto.add(num)
print(lotto)