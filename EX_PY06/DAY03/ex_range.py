#--------------------------------------------------------------
# 내장함수 range()
# - 숫자 범위를 생성하는 내장함수
# - 형식 : range(시작숫자, 끝숫자+1,간격)
#--------------------------------------------------------------
nums=range(1,101)
print(type(nums),len(nums))

# 원소/요소 읽기 => 인덱싱
print(nums[0], nums[-1])

# 원소/요소 여러개 읽기 => 슬라이싱
print(nums[:10], nums[30:41])

# 원소/요소 하나하나를 보기 => list / tuple 형변환
print(list(nums[:10]), tuple(nums[30:41]))

#[실습1] 1~100에서 3의 배수만 저장하세요.
three=range(3,101,3)

print(list(three))

#[실습2] 1.0~10.0 사이에 숫자 사용?
datas=list(range(1,11))
datas=list(map(float,datas))
print(datas)