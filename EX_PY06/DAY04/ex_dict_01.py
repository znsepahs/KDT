#--------------------------------------------------------------------
# Dict 자료형 살펴보기
# - 데이터의 의미를 함께 저장하는 자료형
# - 형태 : {키1:값, 키2:값,...,키n:값}
# - 키는 중복 x, 값은 중복 가능
# - 데이터 분석 시 파일 데이터 가져 올 때 많이 사용
#--------------------------------------------------------------------
## [Dict 생성]
data={}
print(len(data),type(data))

# 사람에 대한 정보 : 이름, 나이, 성별
data={'name':'마징가', 'age':100, 'gender':'남'}
print(len(data),type(data),data)

# 강아지에 대한 정보 : 품종, 무게, 색상, 성별, 나이
data={'cultivar':'비숑', 'weight':'3kg', 'color': 'white','gender':'수컷','age':2}
print(len(data),type(data),data)

# [Dint 원소/요소 읽기]
# - 키를 가지고 값/데이터 읽기
# - 형식: 변수명[키]

# 색상 출력
print(f'색상 : {data["color"]}')

# 성별, 품종 출력
print(f'성별 : {data["gender"]}, 품종 : {data["cultivar"]}')

# [Dint 원소/요소 변경]
# - 변수명[키] = 새로운 값

# 나이 => 6
data['age']=6
print(data)

# weight => 8
data['weight']='8kg'
print(data)

# del 변수명[키] 또는 dal(변수명[키])
del data['gender']
print(data)

# 변수명[새로운 키]=값
data['name']="뽀삐"
print(data)

data['name']="초코"
print(data)