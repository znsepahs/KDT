#--------------------------------------------------------------
# Tuple 데이터 자료형 살펴보기
# - 다양한 종류의 여러개 데이터를 저장하는 타입
# - List와 비슷하지만 수정, 삭제 안됨
# - 형식 : (데1,데2,...,데n) / 데1,데2,...,데n / (데1,) 또는 데,
#--------------------------------------------------------------
# 튜플 데이터 생성
datas=()
print(type(datas), datas, len(datas))

datas=(1,5,7)
print(type(datas), datas, len(datas))

datas=(1,)
print(type(datas), datas, len(datas))

datas=1,
print(type(datas), datas, len(datas))

# 튜플 데이터 원소/요소 읽기
datas=11,22,33,44,55
print(datas[2])

# 1,2,3번 요소 읽기
print(datas[:4])

# 요소/원소 수정 및 삭제 즉, 변경 불가
# 마지막 원소를 'A'로 변경?
# datas[-1]='A' 불가능

# 마지막 원소를 삭제?
# del(datas[-1]) 불가능

# 튜플 데이터 원소/요소 변경 => 형변환
birth=(2024,1,1)

# 1월 => 3월 변경하기
birth=list(birth)
birth[1]=3
print(birth)

# List => tuple 형변환
birth=tuple(birth)