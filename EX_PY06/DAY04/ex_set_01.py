#--------------------------------------------------------------------
# Set 자료형 살펴보기
# - 여러가지 종류의 여러 개 데이터를 저장
# - 단! 중복 안됨
# - 컬렉션 타입의 데이터 저장 시 Tuple가능
# - 형태 : {데이터1, 데이터2, ... ,데이터n}
#--------------------------------------------------------------------
# [Set 생성]
data=set() # 빈 Set. datas={}는 빈 딕셔너리
print(f'data의 타입 : {type(data)}, 원소/요소 개수 : {len(data)}개, 데이터 : {data}')

# 여러개 데이터 저장한 set
data={10,30,20,-9,10,30,10,30,10,10}
print(f'data의 타입 : {type(data)}, 원소/요소 개수 : {len(data)}개, 데이터 : {data}')

data={9.34, 'Apple',10,True,'10'}
print(f'data의 타입 : {type(data)}, 원소/요소 개수 : {len(data)}개, 데이터 : {data}')

# data={1,2,3,[1,2,3]} 리스트는 불가능. 변경이 가능하기 떄문
# data={1,2,3,(1,2,3)} 뉴플은 가능.
# data2={1,2,3,{1:100}} 딕셔너리 불가능.
data={1,2,3,(1,)[0]}
print(f'data의 타입 : {type(data)}, 원소/요소 개수 : {len(data)}개, 데이터 : {data}')

# Set() 내장함수
data={1,2,3} # => set([1,2,3])
data=set() #Empty set
data=set({1,2,3})
# 다양한 타입 => Set 변환
data=set([1,2,1,2,3])
data=set('Good')
print(data,type(data))

# Set 데이터 컨트롤을 위해서는 전용 메서드를 이용
