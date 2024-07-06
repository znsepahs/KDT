#--------------------------------------------------------------------------
# 사용자 정의 함수
#--------------------------------------------------------------------------
# 덧셈, 뺼셈, 곱셈, 나눗셈 함수를 각각 만들기
# - 매개변수 : 정수 2개 => num1, num2
# - 함수결과 : 연산 결과 반환
#--------------------------------------------------------------------------
# 함수 기능 : 2개의 정수로 사칙연산 해주는 함수
# 함수 이름 : ahn
# 매개 변수 : 2개 num1, num2
# 함수 결과 : 연산 결과 반환
#--------------------------------------------------------------------------
def ahnp(num1,num2):
    return num1+num2

def ahnm(num1,num2):
    return num1-num2

def ahnmp(num1,num2):
    return num1*num2

def ahnd(num1,num2):
    return num1/num2 if num2 else "0으로 나눌 수 없음"
#--------------------------------------------------------------------------
# 함수 기능 : 입력 데이터가 유효한 데이터인지 검사해주는 기능
# 함수 이름 : check_data
# 매개 변수 : 문자열 데이터, 데이터 갯수 data, count, sep=' '
# 함수 결과 : 유효 여부 True/False
#--------------------------------------------------------------------------
def check_data(data,count,sep=' '):
    # 데이터 여부
    if len(data):
        # 데이터 분리 후 갯수 체크
        data2=data.split(sep)
        return True if count == len(data2) else False
    else:
        return False
print(check_data('+ 10 3',3))
print(check_data('+ 10',3))
print(check_data('+,10,3',3,','))
    
# [실습] 사용자로부터 연산자, 숫자1, 숫자2를 입력 받아서 연산 결과를 출력해주세요.
data=input().split()
if data[0] not in ['+','-','*','/']:
    print(f'{data[0]} 잘못된 연산자 입니다.')
else:
    if data[1].isdecimal() and data[2].isdecimal():
        if data[0]=='+':print(ahnp(int(data[1]),int(data[2])))
        elif data[0]=='-':print(ahnm(int(data[1]),int(data[2])))
        elif data[0]=='*':print(ahnmp(int(data[1]),int(data[2])))
        elif data[0]=='/':print(ahnd(int(data[1]),int(data[2])))
    else: print("정수만 입력 가능합니다.")