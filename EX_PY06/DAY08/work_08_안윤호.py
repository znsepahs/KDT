#24장 29장 30장
#24.4 연습문제: 파일 경로에서 파일명만 가져오기
path='C:\\Users\\dojang\\AppData\\Local\\Programs\\Python\\Python36-32\\Python.exe'
p1=path.split('.')
filename=p1[-1]
print(filename)
#-----------------------------------------------------------------------------------
#24.5 심사문제: 특정 단어 개수 세기 'the' 갯수 찾아서 출력
txt=input().split()
t1=0
for _ in txt:
    if len(_)==3 and 'the' in _:
        t1=t1+1
    elif 'the.' in _:
        t1=t1+1
print(t1)
#-----------------------------------------------------------------------------------
#24.6 심사문제: 높은 가격순으로 출력하기
price=input().split(';')
p1=[int(a) for a in price ]
p1.sort(reverse=True)
for p2 in p1:
    print(p2)
#-----------------------------------------------------------------------------------
#29.7 연습문제: 몫과 나머지를 구하는 함수 만들기
# x=10
# y=3
# def time_ext(a,b):
#     return a//b,a%b
# time, ext=time_ext(x,y)
# print('몫:{0}, 나머지:{1}'.format(time,ext))
#-----------------------------------------------------------------------------------
#29.8 심사문제: 사칙 연산 함수 만들기
x,y=map(int,input().split())
def calc(num1,num2):
    return num1+num2, num1-num2, num1*num2, num1/num2

a, s, m, d = calc(x,y)
print('덧셈: {0}, 뺄셈: {1}, 곱셈:{2}, 나눗셈:{3}'.format(a, s, m, d))
#-----------------------------------------------------------------------------------
#30.6 연습문제: 가장 높은 점수를 구하는 함수 만들기
# korean, english, mathmatics, science = 100,86,81,91
# def maxsc(*ranscore):
#     return max(ranscore)
# max_score=maxsc(korean,english,mathmatics,science)
# print('높은 점수:',max_score)

# max_score=maxsc(english,science)
# print('높은 점수:',max_score)
#-----------------------------------------------------------------------------------
#30.7 심사문제: 가장 낮은 점수, 높은 점수와 평균 점수를 구하는 함수 만들기
korean, english, mathmatics, science = map(int, input().split())

def get_min_max_score(*ranscore):
    return min(ranscore),max(ranscore)
def get_average(**ranscore):
    cnt=0
    for _ in ranscore.values():cnt=cnt+_
    return cnt/len(ranscore)

min_score, max_score = get_min_max_score(korean, english, mathmatics,science)
average_score = get_average(korean=korean,english=english,mathmatics=mathmatics,science=science)
print('낮은 점수:{0:.2f}, 높은 점수:{1:.2f}, 평균 점수: {2:.2f}'.format(min_score,max_score,average_score))

min_score, max_score = get_min_max_score(english, science)
average_score = get_average(english=english,science=science)
print('낮은 점수:{0:.2f}, 높은 점수:{1:.2f}, 평균 점수: {2:.2f}'.format(min_score,max_score,average_score))

