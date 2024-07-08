#--------------------------------------------------------------------------
# 함수(Function) 이해 및 활용
# 함수기반 계산기 프로그램
# - 4칙 연산 기능별 함수 생성 => 덧셈, 뺄셈, 곱셈, 나눗셈
# - 2개 정수만 계산
#--------------------------------------------------------------------------
# 4칙 연산 통합
def f_opr(opr,num1,num2):
    if opr=='+': print(f'{num1}+{num2}={num1+num2}')
    elif opr=='-': print(f'{num1}-{num2}={num1-num2}')
    elif opr=='*': print(f'{num1}*{num2}={num1*num2}')
    elif opr=='/':
        if not num2:
            print("0이 아닌 숫자만 나눌 수 있습니다.")
        else:
            print(f'{num1}/{num2}={num1/num2}')

f_opr(23,0,"/")

# 4칙 연산 기능별
def ahnp(num1,num2):
    return num1+num2

def ahnm(num1,num2):
    return num1-num2

def ahnmp(num1,num2):
    return num1*num2

def ahnd(num1,num2):
    return num1/num2 if num2 else "0으로 나눌 수 없음"

## 계산기 프로그램
# - 사용자가 종료를 원할때 종료 => 'x', 'X' 입력 시
# - 연산방식과 숫자 데이터 입력 받기
while True:
    #(1) 입력 받기
    req=input("연산(+,-,*,/)방식과 정수 2개 입력(예: + 10 2) :")
    #(2) 종료 조건 검사
    if req=='x' or req=='X':
        print("계산을 종료합니다.")
        break
    #(3) 입력에 연산방식과 데이터 추출'+ 10 2'
    op, num1, num2 = req.split() #['+','10','2']
    num1=int(num1)
    num2=int(num2)
    if op=="+": print(f'{num1}+{num2}={ahnp(num1,num2)}')
    elif op=="-": print(f'{num1}-{num2}={ahnm(num1,num2)}')
    elif op=="*": print(f'{num1}*{num2}={ahnmp(num1,num2)}')
    elif op=="/": print(f'{num1}/{num2}={ahnd(num1,num2)}')
    else: print(f'{op}는 지원되지 않는 연산입니다.')