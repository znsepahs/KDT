#------------------------------------------------------------------------------------------------------------------
# [구현 내용]
# 윷가락은 4개의 값을 저장할 수 있도록 sticks=[0, 0, 0, 0] 형태로 구현
# 윷을 던질 때 마다 랜덤하게 0, 1 사이의 값을 생성해서 sticks[]에 저장하고 점수를 계산함 (예: sticks[i]=random.randint(0,1))	
# 한 명의 점수가 먼저 20점 이상이면 게임은 바로 종료
# '모'나 '윷'이 나온 경우, 이미 총 점수가 20점 이상이면 한 번 더 던지지 않음
# 경기 시작은 어느 누구나 상관없음
# 게임이 종료되면 승패 결과를 화면에 출력하고 프로그램 종료
#------------------------------------------------------------------------------------------------------------------
# [윷놀이 룰]
# 모: 점수 5점 / 배열 [1,1,1,1] - 모두 1
# 윷: 점수 4점 / 배열 [0,0,0,0] - 모두 0
# 걸: 점수 3점 / 배열 [1,0,0,0] - 1이 하나
# 개: 점수 2점 / 배열 [1,1,0,0] - 1이 둘
# 도: 점수 1점 / 배열 [1,1,1,0] - 1이 셋
#------------------------------------------------------------------------------------------------------------------
# [개발자 노트]
# 기능 구현에 랜덤이 필요하니 임포트
# 랜덤으로 윷가락 생성하는 사용자 함수 제작
# 게임을 작동시킬 사용자 함수 제작
# 과제에서 제시한 실행 결과를 반영할 요소들을 윷가락 생성 함수에서 리턴값으로 받아와야 함
# 윷가락 생성 함수 리턴값 : 랜덤으로 생성된 윷가락을 시각적으로 보여줄 리스트, 조합 결과의 이름, 조합 결과의 점수
# while 문 이용하여 무한 반복문으로 일단 설정
# 종료 조건은 둘 중 한명이 20점 넘기는 경우
# 불린 값을 준 플래그 변수로 반복문 제어하여 서로 턴 주고 받게하기
# '모'나 '윷'이 나오면 턴을 유지하게 설정
# 만일 한 플레이어가 '모'혹은 '윷'이 연속으로 나온다면 상대 플레이어는 턴을 얻지 못하고 끝나는지 검증
# 검증용 [1,1,1,1]만 생성하는 사용자 함수 추가 생성
#------------------------------------------------------------------------------------------------------------------

import random as rad

# 윷가락 랜덤 생성 사용자 함수(일반)
def option():
    sticks=[rad.randint(0, 1) for _ in range(4)]  # 윷가락 랜덤 생성
    count=sticks.count(1)
    score=[]
    if count == 4:
        score=[5,'모']  # 모 :5점
    elif count == 0:
        score=[4,'윷']  # 윷 :4점
    elif count == 1:
        score=[3,'걸']  # 걸 :3점
    elif count == 2:
        score=[2,'개']  # 개 :2점
    elif count == 3:
        score=[1,'도']  # 도 :1점
    return [score, sticks]

# 검증용 윷가락 생성 사용자 함수 : [1,1,1,1]만 생성함
def t_option():
    sticks=[rad.randint(1, 1) for _ in range(4)]  # [1,1,1,1]만 생성
    count=sticks.count(1)
    score=[]
    if count == 4:
        score=[5,'모']  # 모 :5점
    elif count == 0:
        score=[4,'윷']  # 윷 :4점
    elif count == 1:
        score=[3,'걸']  # 걸 :3점
    elif count == 2:
        score=[2,'개']  # 개 :2점
    elif count == 3:
        score=[1,'도']  # 도 :1점
    return [score, sticks]

# 게임 실행 사용자 함수
def play_game():
    p1_score = 0
    p2_score = 0
    times=[] # 랜덤 윷가락 조합 저장해놓을 변수. 사용자 함수에서 리스트로 리턴 값을 받아오기 떄문에 빈 리스트
    
    turn=True  # True: 흥부 turn, False 놀부 turn / flag변수 개념
    
    while p1_score < 20 and p2_score < 20:
        if turn :
            times=option()
            sticks=times[1]
            score=int(times[0][0])
            yut=str(times[0][1])
            p1_score += score
            print(f"흥부 {sticks}:{yut} ({score}점) / (총 {p1_score}점)--->")
            if score >= 4:
                if p1_score >= 20:break # 점수가 20점 이상이면 break
                continue  # 모(5점) or 윷(4점)이면 한번 더
            turn = False  # 놀부로 턴 변경
        else:
            times=option()
            sticks=times[1]
            score=int(times[0][0])
            yut=str(times[0][1])
            p2_score += score
            print(f"<---놀부 {sticks}:{yut} ({score}점) / (총 {p2_score}점)")
            if score >= 4:
                if p2_score >= 20:break
                continue  # 모(5점) or 윷(4점)이면 한번 더
            turn = True  # 흥부로 턴 변경

    if p1_score >= 20:print(f'흥부 승리 => 흥부: {p1_score},놀부: {p2_score}')
    else:print(f'놀부 승리 => 흥부: {p1_score},놀부: {p2_score}')

play_game()

#------------------------------------------------------------------------------------------------------------------
# 만일 한 플레이어가 '모'혹은 '윷'이 연속으로 나온다면 상대 플레이어는 턴을 얻지 못하고 끝나는지 검증
#------------------------------------------------------------------------------------------------------------------
def T_play_game():
    p1_score = 0
    p2_score = 0
    times=[] # 랜덤 윷가락 조합 저장해놓을 변수. 사용자 함수에서 리스트로 리턴 값을 받아오기 떄문에 빈 리스트
    
    turn=True  # True: 흥부 turn, False 놀부 turn / flag변수 개념
    
    while p1_score < 20 and p2_score < 20:
        if turn:
            times=t_option() # [1,1,1,1] 윷가락만 생성하는 테스트용 함수 
            sticks=times[1]
            score=int(times[0][0])
            yut=str(times[0][1])
            p1_score += score
            print(f"흥부 {sticks}:{yut} ({score}점) / (총 {p1_score}점)--->")
            if score >= 4:
                if p1_score >= 20:break
                continue  # 모(5점) or 윷(4점)이면 한번 더
            turn = False  # 놀부로 턴 변경
        else:
            times=option()
            sticks=times[1]
            score=int(times[0][0])
            yut=str(times[0][1])
            p2_score += score
            print(f"<---놀부 {sticks}:{yut} ({score}점) / (총 {p2_score}점)")
            if score >= 4:
                if p2_score >= 20:break  # 점수가 20점 이상이면 게임 종료
                continue  # 모(5점) or 윷(4점)이면 한번 더
            turn = True  # 흥부로 턴 변경

    if p1_score >= 20:print(f'흥부 승리 => 흥부: {p1_score},놀부: {p2_score}')
    else:print(f'놀부 승리 => 흥부: {p1_score},놀부: {p2_score}')

#T_play_game()

# 102라인을 주석 처리하고 141라인 주석 처리를 풀어서 실행하면 검증 가능
# 정상 작동함. 흥부에게는 테스트용으로 [1,1,1,1]만 생성하게끔 사용자 함수를 변경하여 주었고
# 놀부는 랜덤으로 윷가락을 생성하는 사용자 함수를 주었음.
# 작동 시키면 놀부는 턴을 얻지 못하고 흥부가 승리