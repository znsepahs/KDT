# -----------------------------------------------------------------
# 누끼 따는 코드
# -----------------------------------------------------------------
from rembg import remove
from PIL import Image
input_path = './image/apple/Apple_Healthy/FreshApple (1).jpg' # 배경 제거할 이미지 경로
output_path = './image/apple/Apple_nugied/FreshApple (1).jpg' # 저장할 이미지 경로

img = Image.open(input_path)
out = remove(img)

# JPEG는 알파 채널을 지원하지 않으므로 RGB로 변환
if out.mode == 'RGBA':
    out = out.convert('RGB')

out.save(output_path)


# -----------------------------------------------------------------
# 딥러닝 모델 돌릴 시에 메모리 확인 하는 코드
# -----------------------------------------------------------------
import psutil

def checkMemory():
    # 현재 프로세스 가져오기
    process = psutil.Process()

    # 메모리 사용량 (바이트 단위)
    memory_info = process.memory_info()

    # Resident Set Size (rss): 실제 사용 중인 물리적 메모리
    memory_usage = memory_info.rss

    # 메모리 사용량을 MB 단위로 출력
    print(f"현재 메모리 사용량: {memory_usage / 1024 ** 2:.2f} MB")