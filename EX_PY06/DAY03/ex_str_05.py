#--------------------------------------------------------------
# 문자열 str 데이터 다루기
# - 이스케이프문자 : 특수한 의미를 가지는 문자
#   * 형식 : \문자1개
#   *'\n' - 줄바꿈문자
#   *'\t' - 탭간격문자
#   *'\'' - 홑따옴표문자
#   *'\"' - 쌍따옴표문자
#   *'\\' - \ 백슬러시 문자, 경로(path), URL관련
#   *'\u' - 유니코드
#   *'\a' - 알람소리
#--------------------------------------------------------------
msg="오늘은 좋은날\n내일은\n주말이라\n행복해"
print(f'줄바꿈 => {msg}')

msg='오늘은 나의 \'생일\'이야'
print(msg)

file='c:\\Users\\LG\\Documents\\test.txt'
print(file)

# r' ' 또는 R' ' : 문자열 내 이스케이프 문자는 무시됨!
file=r'c:\Users\LG\Documents\test.txt'
print(file)

msg="Happy\tNew\tyear"
msg2=R"Happy\tNew\tyear"
print(msg,msg2)