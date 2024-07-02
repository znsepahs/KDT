#--------------------------------------------------------------------
# [실습1] 글자를 입력 받습니다
#         입력 받은 글자의(a~z, A~Z) 코드값을 출력
#--------------------------------------------------------------------
msg=input()
# if msg.isalpha():print(ord(msg))
# else: print("a~z, A~Z 중에서 입력하세요!")

if len(msg) and ('a'<=msg<='z' or 'A'<=msg<='Z'):
    print(f'{msg}의 코드값 {ord(msg)}')
else: print('1개 문자만 입력가능합니다.\n입력 데이터를 확인하세요.')

# 여러개 데이터 입력?
# data="Ab"
# print(list(map(ord, data)))

#--------------------------------------------------------------------
# [실습2] 점수를 입력 받은 후 학점을 출력하세요
# - 학점 : A+,A,A-,B+,B,B-,C+,C,C-,D+,D,D-,F
# A+ : 96~100
# A : 95
# A- : 90~94
#--------------------------------------------------------------------
score=int(input())
if 96<=score<=100: print(f' {score}점 : A+ 학점입니다.')
elif score==95 : print(f' {score}점 : A 학점입니다.')
elif 90<=score<=94 : print(f' {score}점 : A- 학점입니다.')
elif 86<=score<90: print(f' {score}점 : B+ 학점입니다.')
elif score==85 : print(f' {score}점 : B 학점입니다.')
elif 80<=score<=84 : print(f' {score}점 : B- 학점입니다.')
elif 76<=score<80 : print(f' {score}점 : C+ 학점입니다.')
elif score==75 : print(f' {score}점 : C 학점입니다.')
elif 70<=score<=74 : print(f' {score}점 : C- 학점입니다.')
elif 66<=score<70 : print(f' {score}점 : D+ 학점입니다.')
elif score==65 : print(f' {score}점 : D 학점입니다.')
elif 60<=score<=64 : print(f' {score}점 : D- 학점입니다.')
elif score<60 : print(f' {score}점 : F 학점입니다.')
else: print('점수는 0~100점 사이의 값입니다.')