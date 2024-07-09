# 프로그램 : 경주? 경마?
# 출전자 선택을 위한 카탈로그 출력
# 카탈로그 내용 : 출전자들 이름과 특징 설명 / 출전자마다 특징이 있음(예:1번은 초반이 빠르고 후반에 힘 빠짐)
# 출전자 번호를 입력하면 출전자 특징 확인 가능
# 출전자 선택받기
# 출전자 선택받으면 경주 시작
# 구간은 정수값 수치를 가짐, 단위 구간 당 출전자별 특성에따라 이동 할 수 있는 난수범위를 다르게 설정 
# 전체구간은 초/중/후반으로 구분
# 출전자별 특성 구현은 초반이 빠르다고 설명한 출전자는 초반구간 동안(총구간이 300 일때 100 구간까지) 이동거리 난수 범위를 고점에서 좁게 설정하는 식 
# 구간별로 누가 1위인지, 거리를 얼마나 이동했는지 출력
# 결과 발표 후 아무 키나 입력하면 재시작, x 입력하면 프로그램 종료

import random as rad

def run(num1,num2):
    cnt=0
    for t1 in range(100):
        cnt=rad.randint(num1,num2)+cnt
    return cnt

def print_menu():
    print(f'{"▼":▼^19}')
    print(f'{"▶     출전자    ◀":^17}')
    print(f'{"-":-^19}')
    print(f'{"▶     1태웅     ◀":^18}')
    print(f'{"▶     2원석     ◀":^18}')
    print(f'{"▶     3치영     ◀":^18}')
    print(f'{"▶     4알중     ◀":^18}')
    print(f'{"▶     5용병     ◀":^18}')
    print(f'{"-":-^19}')
    print(f'{"▶     6선택     ◀":^18}')
    print(f'{"▲":▲^19}')

def select_menu():
    print(f'{"▼":▼^19}')
    print(f'{"▶     Select    ◀":^19}')
    print(f'{"-":-^19}')
    print(f'{"▶     1태웅     ◀":^18}')
    print(f'{"▶     2원석     ◀":^18}')
    print(f'{"▶     3치영     ◀":^18}')
    print(f'{"▶     4알중     ◀":^18}')
    print(f'{"▶     5용병     ◀":^18}')
    print(f'{"▲":▲^19}')

def delay():
    for _ in range(40000000):pass
    print('Now loading...')
    for _ in range(65000000):pass
    print('...')
    for _ in range(75000000):pass
    print('Complete')

isBreak=False
while True:
    if isBreak: break
    while True:
        print_menu()
        info=input("선수번호를 입력하면 특징을 알 수 있습니다. 경기를 시작하려면 6을 입력하세요.")

        if info=='1':
            print("'태웅'은 초반에 매우 빠르고 후반에 느려집니다.")
        elif info=='2':
            print("'원석'은 경기 중반부터 가속이 됩니다.")
        elif info=='3':
            print("'치영'은 후반부의 매우 빠릅니다.")
        elif info=='4':
            print("'알중'은 전 구간에서 기복이 심합니다.")
        elif info=='5':
            print("'용병'은 모든것이 수수께끼인 인물입니다.")
        elif info=='6':
            print("선수를 선택하고 경기를 시작합니다.")
            break
        else: print("존재하지 않는 항목입니다. 입력을 확인해주세요.")

    while True:
        select_menu()
        info=input("선수번호를 선택하면 경기가 시작됩니다.")

        if info=='1':
            print("'1태웅'을 선택하셨습니다. 경기를 시작합니다.")
            break
        elif info=='2':
            print("'2원석'을 선택하셨습니다. 경기를 시작합니다.")
            break
        elif info=='3':
            print("'3치영'을 선택하셨습니다. 경기를 시작합니다.")
            break
        elif info=='4':
            print("'4알중'을 선택하셨습니다. 경기를 시작합니다.")
            break
        elif info=='5':
            print("'5용병'을 선택하셨습니다. 경기를 시작합니다.")
            break
        else: print("존재하지 않는 항목입니다. 입력을 확인해주세요.")


    delay()
    print('▶'*105)
    r1={"1태웅":run(66,100),"2원석":run(1,34),"3치영":run(10,34),"4알중":run(3,90),"5용병":run(50,60)}
    print(f' 초반부 점수 : {sorted(r1.items(),key=lambda x:x[1],reverse=True)}')
    print('▶'*105)
    delay()
    print('▶'*105)
    r2={"1태웅":run(33,60),"2원석":run(60,80),"3치영":run(36,54),"4알중":run(5,87),"5용병":run(31,56)}
    print(f' 중반부 점수 : {sorted(r2.items(),key=lambda x:x[1],reverse=True)}')
    print('▶'*105)
    r1v=list(r1.values())
    r2v=list(r2.values())
    fmp={"1태웅":(r1v[0]+r2v[0]),"2원석":(r1v[1]+r2v[1]),"3치영":(r1v[2]+r2v[2]),"4알중":(r1v[3]+r2v[3]),"5용병":(r1v[4]+r2v[4])}
    print(f' 중반부 결산 : {sorted(fmp.items(),key=lambda x:x[1],reverse=True)}')
    print('▶'*105)
    delay()
    print('▶'*105)
    r3={"1태웅":run(1,34),"2원석":run(50,70),"3치영":run(70,90),"4알중":run(10,92),"5용병":run(22,72)}
    print(f' 후반부 점수 : {sorted(r3.items(),key=lambda x:x[1],reverse=True)}')
    print('▶'*105)
    fmpv=list(fmp.values())
    r3v=list(r3.values())
    finalp={"1태웅":(fmpv[0]+r3v[0]),"2원석":(fmpv[1]+r3v[1]),"3치영":(fmpv[2]+r3v[2]),"4알중":(fmpv[3]+r3v[3]),"5용병":(fmpv[4]+r3v[4])}
    print('♣'*105)
    print(f' 최종 결산 : {sorted(finalp.items(),key=lambda x:x[1],reverse=True)}')
    print('♣'*105)
    fp=sorted(finalp.items(),key=lambda x:x[1],reverse=True)
    if info in fp[0][0]: print("★축하합니다★ 선택한 출전자가 우승했습니다!")
    else : print("아쉽네요 선택한 출전자의 컨디션이 좋지 않았나 봅니다~")

    restrt=input("한번더 플레이 하시려면 아무 키나 누르세요. 'x'을 입력하면 프로그램을 종료합니다.").strip()
    if restrt=='x': isBreak=True
    else : print("처음화면으로 돌아갑니다.")