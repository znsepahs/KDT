#--------------------------------------------------------------------
# Dict 자료형 살펴보기
# - 연산자와 내장함수
#--------------------------------------------------------------------
person={'name': '홍길동', 'age':20, 'job': '학생'}
dog={'cultivar':'비숑', 'weight':'3kg', 'color': 'white','gender':'수컷','age':2}
jumsu={'국어':90,'수학':178,'체육':100,}

# [연산자]
# 산술 연산 x
#person+dog?

# 멤버 연산자 : in, not in
print('name' in dog)
print('name' in person)

# value : dict 타입에서는 key만 멤버 연산자로 확인
#print('허스키' in dog)
#print(20 in person)

# value 추출 메서드 values() 사용
print('허스키' in dog.values())
print(20 in person.values())

# [내장함수]
# - 원소/요소 개수 확인
print(f' dog의 요소 개수 : {len(dog)}개')
print(f' person의 요소 개수 : {len(person)}개')

# - 원소/요소 정렬 : sorted()
# - 키만 정렬
print(f' dog 오름차순정렬 : {sorted(dog)}')
print(f' dog 내림차순정렬 : {sorted(dog,reverse=True)}')

print(f' jumsu 값 오름차순정렬 : {sorted(jumsu.values())}')
print(f' jumsu 키 오름차순정렬 : {sorted(jumsu)}')

print(f' jumsu 값 오름차순정렬 : {sorted(jumsu.items())}')
print(f' jumsu 값 오름차순정렬 : {sorted(jumsu.items(),key=lambda x:x[1])}')
print(f' jumsu 값 내림차순정렬 : {sorted(jumsu.items(),key=lambda x:x[1],reverse=True)}')