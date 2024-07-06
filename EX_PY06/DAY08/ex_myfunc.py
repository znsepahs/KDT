#---------------------------------------------------------------------
# 내가 만든 함수들
#---------------------------------------------------------------------
def add(num1,num2): return num1+num2

print(f'__name__: {__name__}')

if __name__ == '__main__':
    print("--TEST")
    print(f'결과:{add(100,100)}')

#입력데이터 체크용--------------------------------------------------------------------
def check_data(data,count,sep=' '):
    # 데이터 여부
    if len(data):
        # 데이터 분리 후 갯수 체크
        data2=data.split(sep)
        return True if count == len(data2) else False
    else:
        return False
#print(check_data('+ 10 3',3))

#사칙연산용---------------------------------------------------------------------------
def ahnp(num1,num2):
    return num1+num2

def ahnm(num1,num2):
    return num1-num2

def ahnmp(num1,num2):
    return num1*num2

def ahnd(num1,num2):
    return num1/num2 if num2 else "0으로 나눌 수 없음"
