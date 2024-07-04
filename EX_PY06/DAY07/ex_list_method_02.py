#---------------------------------------------------------------------
# 리스트 전용의 함수 즉, 메서드(method)
# - 리스트의 원소/요소를 제어하기 위한 함수들
#---------------------------------------------------------------------
# [메서드 - 요소 추가 메서드 append(데이터)]
datas=[1,3,5]

#새로운 데이터 100 추가 : 제일 마지막 원소로 추가
datas.append(100)
print(f'datas의 개수 : {datas}, {len(datas)}')

datas.append(100)
print(f'datas의 개수 : {datas}, {len(datas)}')

# [메서드 - 요소 추가 메서드 insert(인덱스, 데이터)]
# 원하는 위치에 새로운 데이터 추가. 지정 인덱스에 추가되고 기존의 요소는 밀림
datas.insert(0,300)
print(f'datas의 개수 : {datas}, {len(datas)}')

datas.insert(-1,300)
print(f'datas의 개수 : {datas}, {len(datas)}')

#[실습] 임의의 정수 숫자 10개 저장하는 리스트 생성
# - random 모듈
# - 빈리스트 생성
# - for 반복문
import random as rad
nums=[]
for cnt in range(10):
    nums.append(rad.randint(1,100))
    
print(nums)

# [메서드 - 요소 삭제 메서드 remove(데이터)]
# 중복시 왼쪽에서 >>>> 오른쪽으로 지워나감. 존재하지 않는 데이터 삭제 시 Error
datas.remove(300)
print(f'datas의 개수 : {datas}, {len(datas)}')

for cnt in range(datas.count(300)):
    datas.remove(300)
    print(f'datas의 개수 : {datas}, {len(datas)}')