# 22장, 25장 320p 까지
# 22.9 연습문제: 리스트에서 특정 요소만 뽑아내기
#요청: ['alpha','bravo','delta','hoteal','india']
a=['alpha','bravo','charlie','delta','echo','foxtrot','golf','hotel','india']
b=[]
c=[]
print(b)
for c in a:
    if len(c)==5:
        b.append(c)
print(b)
print()
#--------------------------------------------------------------------------------
#22.1 심사문제: 2의 거듭제곱 리스트 생성하기
result=[]
int1, int2=map(int,input().split())
if int1<=int2:
    for a in range(int1,int2+1):
        result.append(2**a)
print(result)
print()
#--------------------------------------------------------------------------------
#25.1 딕셔너리 조작
x={'a':10,'b':20,'c':30,'d':40}
x.setdefault('e') #키만 저장하여 값은 None
x.setdefault('f',100) #'f':100 저장
#25.1.3 딕셔너리 키:값 수정하기
x={'a':10,'b':20,'c':30,'d':40}
x.update(a=90) #'a'의 값을 90으로 변경
x.update(e=50) #'e':50 저장
x.update(a=900,f=60) #'a'값을 90으로 변경하고 'f':60 저장
y={1:'one',2:'two'} #update()는 키가 문자열일 때만 사용가능
y.update({1:'ONE',3:'THREE'}) #이런 방식의 수정도 가능
y.update([[2,'TWO'],[4,'FOUR']]) #리스트를 이용한 값 수정
y.update(zip([1,2],['one','two'])) #zip 사용해서 값 수정
#25.1.4 딕셔너리 키:값 삭제하기
x={'a':10,'b':20,'c':30,'d':40}
x.pop('a') #딕셔너리에서 키:값을 삭제한 뒤 삭제된 값을 반환, 키가 없으면 디폴트값 0 반환
#25.1.5 딕셔너리에서 임의의 키:값 쌍 삭제하기
x={'a':10,'b':20,'c':30,'d':40}
x.popitem() #파이썬 버전3.9로 실행하여 'd':40을 삭제하고 튜플로 반환
#25.1.6 딕셔너리 모든 키:값 쌍을 삭제
x={'a':10,'b':20,'c':30,'d':40}
x.clear()
#25.17 딕셔너리에서 키의 값을 가져오기
x={'a':10,'b':20,'c':30,'d':40}
x.get('a') #'a'의 값 10 반환 / 존재하지 않는 키에 사용하면 디폴트값 0을 반환
#25.18 딕셔너리에서 키:값 쌍을 모두 가져오기
x={'a':10,'b':20,'c':30,'d':40}
# x.items() x.keys() x.(values) #가장 중요한 메서드
#25.1.9 리스트와 튜플로 딕셔너리 만들기
keys=['a','b','c','d']
x=dict.fromkeys(keys) #딕셔너리 생성, 값은 모두 None
y=dict.fromkeys(keys,100) #y={'a':100,'b':100,'c':100,'d':100}
#25.2 반복문으로 딕셔너리의 키:값 쌍을 모두 출력하기
x={'a':10,'b':20,'c':30,'d':40}
for key, value in x.items(): print(key,value)
#25.2.1 딕셔너리의 키만 출력하기
x={'a':10,'b':20,'c':30,'d':40}
for key in x.keys(): print(key)
#25.2. 딕셔너리의 값만 출력하기
x={'a':10,'b':20,'c':30,'d':40}
for value in x.values(): print(value)