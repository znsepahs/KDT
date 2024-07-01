#-----------------------------------------------------------------------------
# 연산자 실습
#-----------------------------------------------------------------------------
#[실습] 문자열 데이터 2개에 대한 논리 연산 (and, or) 수행 후 출력
data1='Hello'
data2='hello'
print(f'{data1}>{data2} and {data1}=={data2} : {data1>data2 and data2==data2}')
print(f'{data1}<{data2} or {data1}=={data2} : {data1<data2 or data2==data2}')

#[실습] 정수1개와 문자열 1개에 대한 논리 연산(not) 수행 후 출력
#num=-3.2 , 0인 경우
#msg='원더우먼', ''인 경우
num=-3.2
msg='원더우먼'
print(f'not {num} : {not num}')
print(f'not {msg} : {not msg}')

num=0 # False
msg='' # False
print(f'not {num} : {not num}')
print(f'not {msg} : {not msg}')