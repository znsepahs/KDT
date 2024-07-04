#[18.5] continue
# 0~73 사이의 숫자 중 3으로 끝나는 숫자 출력

i=0
while True:
    if i%10!=3:
        i=i+1
        continue
    if i>73:break
    print(i,end=' ')
    i=i+1

#[Unit.19]별 출력하기
## [1] outer=5, inner=5
for i in range(5):
    for j in range(5):
        print(f'j:{j}',end=' ')
    print(f'i:{i}\\n')

## [2] 대각선 * 출력
# *
#  *
#   *
#    *
#     *
for v in range(5):
    for h in range(v+1):
        # if h ==v:
        #     print('*')
        # else:print(' ',end='')
        print('*' if h==v else ' ',end='\n' if h==v else '')
