#16.4퀴즈
##1. 다음 중 for 로 10번 반복하는 방법으로 올바른 것을 모두 고르세요.
#정답: a,d
#b:11회 , e:9회

##2. 다음 중 20부터 10까지 출력하는 방법으로 올바른 것을 모두 고르세요.
#정답: c,d
#a:공백 , b:공백, e:10~19내림차순

##3. 다음 소스 코드에서 잘못된 부분을 모두 고르세요.
#a. count = input()
#b. 
#c. for i in range(count)
#d.     print('i의 값은',end=' ')
#e.     print(i)
#정답: a - 입력받은 str을 c에서 활용하려면 int()를 사용해야함
#      d,e - 한 줄로 짤 수 있는데 굳이 2줄을 사용
#            수정한다면 print('i의 값은, i)

##4. 다음 for 반복문을 실행했을 떄의 출력 결과를 고르세요.
for i in reversed('Python'):
    print(i, end='.')
#정답: d - 문자열 'Python'의 마지막 인덱스 부터 가져와 i에 넣고 프린트, 줄바꿈이 디폴트인 end를 '.'로 변경
#----------------------------------------------------------------------------------------------------
#16.5연습문제: 리스트의 요소에 10을 곱해서 출력하기. 숫자는 공백으로 구분하고 한 줄로 출력
x=[49,-17,25,102,8,62,21]
for r in x:
    print(r*10,end=' ')
#----------------------------------------------------------------------------------------------------
#16.6심사문제: 입력받은 정수의 구구단을 출력하는 프로그램을 만드세요.
gugu=int(input())
for num in range(1,10):
    print(f'{gugu} * {num} = {gugu*num}')