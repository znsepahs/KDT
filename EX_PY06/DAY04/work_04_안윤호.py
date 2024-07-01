#12.5 심사문제: 딕셔너리에 게임 캐릭터 능력치 저장하기
lol1=input().split()
lol2=input().split()
lol=dict(zip(lol1,lol2))
print(lol)
#---------------------------------------------------------------------
#13.6 연습문제: if 조건문 사용하기
x=5
if x !=10:print('OK')
#---------------------------------------------------------------------
#13.7 심사문제: 온라인 할인 쿠폰 시스템 만들기
price=int(input())
cupon=input()
if cupon in 'cash3000': print(f'{price-3000}')
elif cupon in 'cash5000': print(f'{price-5000}')
else: print('쿠폰이름을 제대로 입력해주세요!')
#---------------------------------------------------------------------
#14.6 연습문제: 합격 여부 판단하기
written_test=75
coding_test=True
if written_test >=80 and coding_test==True: print('합격')
else: print('불합격')
#---------------------------------------------------------------------
#14.7 심사문제: 합격 여부 판단하기
score=input().split()
if int(score[0]) in range(101) and int(score[1]) in range(101) and int(score[2]) in range(101) and int(score[3]) in range(101):
    if (int(score[0])+int(score[1])+int(score[2])+int(score[3])/4)>=80: print('합격')
    else: print('불합격')
else: print('잘못된 점수')
#---------------------------------------------------------------------
#15.3 연습문제: if,elif,else 모두 사용하기
x=int(input())
if x>=11 and x<20:
    print('11~20')
elif x>=21 and x<30:
    print('21~30')
else: print('아무것도 해당하지 않음')
#---------------------------------------------------------------------
#15.4 심사문제: 교통카드 시스템 만들기
# 어린이(초등학생,만 7세 이상 12세 이하):650원
# 청소년(중고등학생,만 13세 이상 18세 이하):1050원
# 어른(일반, 만 19세 이상):1250원
# 교통카드 잔액 9000원
age=int(input())
balance=9000
if 7<=age<13: balance=balance-650
elif 13<=age<19: balance=balance-1050
else: balance=balance-1250
print(balance)
#---------------------------------------------------------------------