import pandas as pd

d1={'Seoul':['South Korea','Asia','9655000'],
    'Tokyo':['Japan','Asia','14110000'],
    'Beijing':['China','Asia','21540000'],
    'London':['United Kingdom','Europe','14800000'],
    'Barlin':['Germany','Europe','3426000'],
    'Mexico City':['Mexico','America','21200000']}

df=pd.DataFrame(d1)

def print_menu():
    print(f'{"▼":▼^19}')
    print(f'{"▲":▲^19}')

isBreak=False
while True:
    if isBreak: break
    while True:
        print_menu()
        info=input("메뉴를 입력하세요.")

        if info=='1':
            print(df)
        elif info=='2':
            print("'원석'은 경기 중반부터 가속이 됩니다.")
        elif info=='3':
            print("'치영'은 후반부에 매우 빠릅니다.")
        elif info=='4':
            print("'알중'은 전 구간에서 기복이 심합니다.")
        elif info=='5':
            print("'용병'은 모든것이 수수께끼인 인물입니다.")
        elif info=='6':
            print("선수를 선택하고 경기를 시작합니다.")
            break
        else: print("존재하지 않는 항목입니다. 입력을 확인해주세요.")

    while True:
        info=input("메뉴를 입력하세요.")

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


    print('▶'*105)
    r1={"1태웅":run(66,100),"2원석":run(1,34),"3치영":run(10,34),"4알중":run(3,90),"5용병":run(50,60)}
    print(f' 초반부 점수 : {sorted(r1.items(),key=lambda x:x[1],reverse=True)}')
    print('▶'*105)
    print('▶'*105)
    r2={"1태웅":run(33,60),"2원석":run(60,80),"3치영":run(36,54),"4알중":run(5,87),"5용병":run(31,56)}
    print(f' 중반부 점수 : {sorted(r2.items(),key=lambda x:x[1],reverse=True)}')
    print('▶'*105)
    r1v=list(r1.values())
    r2v=list(r2.values())
    fmp={"1태웅":(r1v[0]+r2v[0]),"2원석":(r1v[1]+r2v[1]),"3치영":(r1v[2]+r2v[2]),"4알중":(r1v[3]+r2v[3]),"5용병":(r1v[4]+r2v[4])}
    print(f' 중반부 결산 : {sorted(fmp.items(),key=lambda x:x[1],reverse=True)}')
    print('▶'*105)
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
    if info in fp[0][0]: print("★")
    else : print("아")

    restrt=input("'x'을 입력하면 프로그램을 종료합니다.").strip()
    if restrt=='x': isBreak=True
    else : print("처음화면으로 돌아갑니다.")