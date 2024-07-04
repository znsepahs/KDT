#---------------------------------------------------------------------
# Dict 전용 함수. 메서드(method)
# => keys(),values(),items() 필수!!
#---------------------------------------------------------------------
person={'name':'홍길동','age':10}

# [메서드 - 값 읽어오는 메서드 get(key,default)]
# - key에 해당하는 value가 없으면 default값(None)을 반환
print(person['name'])
# print(person['gender']) # key error 발생

print(person.get('name','몰라'))
print(person.get('gender','없음'))
print(person.get('gender'))

# if gender in key.values:

# [메서드 - 키와 값 추가 메서드]
person['gender']='남'

msg="Hello Good Luck"
alpha=set(msg)
save={}
for m in alpha:
    print(m,msg.count(m))
    save[m]=msg.count(m)
print(save)

# [메서드 - 수정 및 업데이트 메서드 update(k=v)]
# 수정 / 추가 / 업데이트
person['gender']='여'

person.update(gender='어린이')
print(person)

person.update(gender='어린이',job='학생')
print(person)

person.update({'phone':'010','birth':'240101'})
print(person)

# **{'weight':100,'height':170} => weight=100, height=170
person.update(**{'weight':100,'height':170})
print(person)
