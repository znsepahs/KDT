#--------------------------------------------------------------------------
# Flask Framework에서 WebServer 구동 파일
# - 파일명 : app.py
#--------------------------------------------------------------------------
# 모듈로딩
from flask import Flask, render_template

# 전역변수
# Flask Web Server 인스턴스 생성
APP=Flask(__name__)

# 라우팅 기능 함수
# @Flask Web Server 인스턴스변수명.route("URL")
@APP.route("/") # "/" => 루트(root) : SW의 시작 폴더, Linux/Mac 저장소 시작점
def index():
    return render_template("index.html")

@APP.route("/info")
@APP.route("/info/") # 이렇게 하면 두 가지 도메인(/info, /info/)에서 공통으로 작동하는 페이지
def printInfo():
    return render_template("info.html")

# http://127.0.0.1:5000/info/문자열변수
# name에 문자열 변수 저장, name=문자열변수
@APP.route("/info/<name>")
def printInfo2(name):
    return f"""<body style='background-color:coral; text-align:center'>
    <h1>{name}'s INFORMATION</h1>HELLO~"""

# http://127.0.0.1:5000/info/정수
# age라는 변수에 정수 저장
@APP.route("/info/<int:age>")
def checkAge(age):
    return f"""<body style='background-color:pink; text-align:center'>
    나이 : {age}</body>"""

# http://127.0.0.1:5000/go
@APP.route("/go")
def goHome():
    return APP.redirect("/")

# 조건부 실행
if __name__ =='__main__':
    # Flast Web Server 구동
    APP.run()