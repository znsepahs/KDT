#-----------------------------------------------------------------------
# 1줄로 조건식을 축약 : 조건부 표현식
#-----------------------------------------------------------------------
# [실습] 임의의 숫자가 5의 배수 여부 결과를 출력하세요
ran_nums=int(input('숫자입력하세용'))
print('5의배수아님') if ran_nums%5 else print('5의배수')

# [실습] 문자열을 입력 받아서 문자열의 원소 개수를 저장
# - 단 원소 개수가 0이면 None 저장
# - (1) 입력받기 input()
# - (2) 원소/요소 개수 파악 len()
# - (3) 원소/요소 개수 저장 단, 0인경우 None 저장하기
msg=input('문자열을 입력해 주세요')
result=len(msg) if len(msg) else None
print(result)

# [실습] 연산자(4칙연산자 : =,-,*,/)와 숫자 2개 입력 받기
# - 입력된 연산자에 따라 계산 결과 저장
# - 예) 입력 : + 10 3 / 출력 : 13
num=input('연산자와 숫자 2개 입력해주세요 : ').split()
if num[0]=='+':
    result=int(num[1])+int(num[2])
elif num[0]=='-':
    result=int(num[1])-int(num[2])
elif num[0]=='*':
    result=int(num[1])*int(num[2])
elif num[0]=='/':
    result=int(num[1])/int(num[2])
print(result)