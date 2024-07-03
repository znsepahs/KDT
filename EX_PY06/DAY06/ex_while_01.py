#---------------------------------------------------------------------
# 제어문 - while 반복문
#---------------------------------------------------------------------
num=10
while num>0:
    print(num)
    num=num-1

# [실습] 리스트의 원소 읽기
# - while 반복문 : 개수
nums=[11,22,33]

cnt=0
while cnt<len(nums):
    print(nums[cnt])
    cnt=cnt+1

# [실습] "Hello" 문자열의 원소를 하나씩 출력하기
msg="Hello"

time=0
while time<len(msg):
    print(msg[time])
    time=time+1