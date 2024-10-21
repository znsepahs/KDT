#--------------------------------------------------------------------------
# Flask Framework에서 WebServer 구동 파일
# - 파일명 : app.py
#--------------------------------------------------------------------------
# 모듈로딩
from flask import Flask

# 전역변수
# Flask Web Server 인스턴스 생성
APP=Flask(__name__)

# 라우팅 기능 함수
# @Flask Web Server 인스턴스변수명.route("URL")
@APP.route("/") # "/" => 루트(root) : SW의 시작 폴더, Linux/Mac 저장소 시작점
def index():
    return """
    <body style='background-color:green;'
    <h1><marquee behavior>HELLO</marquee behavior></h1>
    </body>"""


# 조건부 실행
if __name__ =='__main__':
    # Flast Web Server 구동
    APP.run()