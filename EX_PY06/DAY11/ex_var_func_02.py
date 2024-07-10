#--------------------------------------------------------------------------
# 함수와 변수 - 지역/전역 변수
#--------------------------------------------------------------------------
## 전역변수(Global Variable)
## - 파일(py) 내에 존재하며 모든 곳에서 사용 가능
## - 프로그램 실행 시 메모리 존재
## - 프로그램 종료 시 메모리에서 삭제
#--------------------------------------------------------------------------
persons=["Hong"]
gender={'H':'남자'}
year=2024

## 사용자 정의 함수
def add_person(name):
    global year
    year +=1
    persons.append(name)
    gender[name]='여'

def remove_person(name):
    persons.remove(name)
    gender.pop(name)

## 실행 즉, 함수 호출
print(f'persons=>{persons}')
print(f'gender=>{gender}')

add_person("Park")
print(f'persons=>{persons}')
print(f'gender=>{gender}')

remove_person("Park")
print(f'persons=>{persons}')
print(f'gender=>{gender}')
