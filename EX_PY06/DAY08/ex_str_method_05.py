#--------------------------------------------------------------------------
# str 데이터 타입 전용 함수 즉, 메서드 살펴보기
#--------------------------------------------------------------------------
# [1개 문자열을 여러개 문자열로 분리해주는 메서드 split()]
# - 분리기준 => 기본값 : 공백
msg=" Happy New Year "
result=msg.split()
print(f'result => {type(result)},{result}')

phone='010-2222-3333'
presult=phone.split('-')
print(f'presult => {type(presult)},{presult}')

msg='오늘은 날씨가 좋군요. 내일도 날씨가 좋을까?'
result=msg.split('.')
print(f'result => {type(result)},{result}')

# [여러개 문자열을 하나의 문자열로 합쳐주는 메서드 join()]
# - 합칠문자.join(여러개 문자열)
# => 010*2222*3333
con='*'
phone2=con.join(presult)
print(f'phone2 => {phone2}')
