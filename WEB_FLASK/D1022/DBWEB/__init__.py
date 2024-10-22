#--------------------------------------------------------------------------
# Flask Framework에서 WebServer 구동 파일
# - 파일명 : app.py
#--------------------------------------------------------------------------
# 모듈로딩
from flask import Flask

#--------------------------------------------------------------------------
# Application 생성 함수
# - 함수명 : create_app <= 이름변경 불가!!!
#--------------------------------------------------------------------------
def create_app():
    # Flask Web Server 인스턴스 생성
    APP=Flask(__name__)

   # URL 처리 모듈 등록
    from .views import main_view
    APP.register_blueprint(main_view.mainBP)

    return APP