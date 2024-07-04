#---------------------------------------------------------------------
# 리스트 전용의 함수 즉, 메서드(method)
# - 리스트의 원소/요소를 제어하기 위한 함수들
#---------------------------------------------------------------------
import random as rad

# [1] 실습 데이터 -> 임의의 정수 숫자 10개 구성된 리스트
datas=[1,5,15,5,92,0,5,10,9,8]

# [메서드 - 요소 인덱스를 반환하는 메서드 index()]
# -> 92의 인덱스를 찾기
# -> 완쪽 >>>> 오른쪽으로 찾음
idx=datas.index(92)
print(f'92의 인덱스 : {idx}')

# -> 리스트에 없는 요소의 인덱스 찾기? 존재하지 않는 데이터의 경우 Error 발생
if 118 in datas:
    idx=datas.index(92)
    print(f'92의 인덱스 : {idx}')
else: print("존재하지 않는 데이터입니다.")

# -> 5의 인덱스 찾기
if 5 in datas:
    idx=datas.index(5)
    print(f'첫번째 5의 인덱스 {idx}')

    idx=datas.index(5,idx+1)
    print(f'두번째 5의 인덱스 {idx}')

    idx=datas.index(5,idx+2)
    print(f'세번째 5의 인덱스 {idx}')

# [메서드 - 데이터가 몇개 존재하는지 파악하는 메서드 count(데이터)]
cnt=datas.count(5)
print(f'5의 개수:{cnt}개')
idx=0
for i in range(cnt):
    idx=datas.index(5,idx if not i else idx+1)
    print(f'{i+1}번째 5의 인덱스 : {idx}')
