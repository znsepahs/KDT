#--------------------------------------------------------------------------
# Flask Framework에서 모듈단위 URL 처리 파일
# - 파일명 : main_view.py
#--------------------------------------------------------------------------
# 모듈로딩
from flask import Blueprint, render_template

# Blueprint 인스턴스 생성
mainBP=Blueprint('MAIN', import_name=__name__, url_prefix='/', template_folder='templates')

# http://localhost:8000/ URL 처리 라우팅 함수 정의
@mainBP.route("/")
def index():
    return render_template('index.html')