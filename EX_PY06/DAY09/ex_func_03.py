# 4칙 연산 기능별
def ahnp(num1,num2):
    return num1+num2

def ahnm(num1,num2):
    return num1-num2

def ahnmp(num1,num2):
    return num1*num2

def ahnd(num1,num2):
    return num1/num2 if num2 else "0으로 나눌 수 없음"

#-----------------------------------------------------------------------------------
# 함수기능 : 계산기 메뉴를 출력하는 함수
# 함수이름 : print_menu
# 매개변수 : 없음
# 함수결과 : None
#-----------------------------------------------------------------------------------
def print_menu():
    print(f'{"*":*^19}')
    print(f'{"계 산 기":^18}')
    print(f'{"*":*^19}')
    print(f'{"* 1   덧    셈   *":^18}')
    print(f'{"* 2   뺄    셈   *":^18}')
    print(f'{"* 3   곱    셈   *":^18}')
    print(f'{"* 4   나 눗 셈   *":^17}')
    print(f'{"* 5   종    료   *":^18}')
    print(f'{"*":*^19}')

#-----------------------------------------------------------------------------------
# 함수기능 : 계산기 메뉴를 출력하는 함수
# 함수이름 : calc
# 매개변수 : 함수명, str 숫자 2개
# 함수결과 : None
#-----------------------------------------------------------------------------------
def calc(func,op):
    num1, num2=input("정수 2개(예: 10 2):").split()
    num1=int(num1)
    num2=int(num2)
    print(f'결과 {num1}{op}{num2}={func(num1,num2)}')

## 계산기 프로그램
# - 사용자에게 원하는 계산을 선택할 메뉴 출력
# - 종료 메뉴 선택 시 프로그램을 종료
# => 무한반복 : while
while True:
    print_menu()
    choice=input("메뉴 선택: ")
    if choice.isdecimal():
        choice=int(choice)
    else :
        print("0~9사이의 숫자만 입력하세욧!")
        continue

    if choice==5:
        print("프로그램을 종료합니다.")
        break
    elif choice==1:
        print("덧셈")
        calc(ahnp,'+')
    elif choice==2:
        print("뺄셈")
        calc(ahnm,'-')
    elif choice==3:
        print("곱셈")
        calc(ahnmp,'*')
    elif choice==4:
        print("나눗셈")
        calc(ahnd,'/')
    else :print("제공하지 않는 기능입니다.")

    
