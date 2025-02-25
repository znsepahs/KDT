#--------------------------------------------------------------
# 내장함수 print() 사용법
# - 모니터 즉, 콘솔/터미널에 출력하는 함수
# - 문법: print(argument,argument1,argument2,...,) / print()
#--------------------------------------------------------------
#나이, 이름, 성별을 저장하기
age=32
name='홍길동'
gender='남'


#모니터에 출력하기
print(age, name, gender, sep='\n')
#한 줄에 모든 변수 출력하기
print(age, name, gender)

# 2개의 정수 덧셈 결과 출력하기
num1=2
num2=9
print(num1+num2)


#서식지정자 (Format string) 방식
#=> 화면 출력 글자를 만들고 그 글자안에 특정결과를 출력하는 형식
#=> 글자 내부에 정수결과 넣기: '%d' %변수명 또는 %수식
#=> 글자 내부에 실수결과 넣기: '%f' %변수명 또는 %수식
#=> 글자 내부에 글자결과 넣기: '%s' %변수명 또는 %수식
#2+9=11 화면에 출력
print('%d+%d=%d' %(num1, num2, num1+num2))

# 9 / 2 = 4.5 화면에 출력
print('%d/%d=%f' %(num2, num1, num2/num1))
print('%s/%s=%s' %(num2, num1, num2/num1))

#F-string 방식
#-형식: f' {변수명 또는 수식 또는 데이터}  '
#-형식: F" {변수명 또는 수식 또는 데이터}  "
# 2 +9 = 11 화면에 출력
print(f'{num1}+{num2}={num1+num2}')
# 9 / 2 = 4.5 화면에 출력
print(f'{num2}/{num1}={num2/num1}')