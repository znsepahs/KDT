#---------------------------------------------------------------------
# 제어문 - while 반복문
#---------------------------------------------------------------------
# [실습] 구구단 3단 while 문 사용하여 출력하기
dan=1
while dan<10:
    print(f'3*{dan}={3*dan}')
    dan=dan+1

# [실습] 1~30 범위의 수 중에서 홀수만 출력
#        단 While문 사용
num=1
while num<31:
    if num%2:
        print(num)
    num=num+1

while num<31:
    print(num)
    num=num+2