#-------------------------------------------------------------------------
# Flask Framework에서 '/'URL에 대한 라우팅 처리 파일
# - 파일명 : main_views.py
#-------------------------------------------------------------------------
# 모듈로딩
from flask import Blueprint, render_template

# Blueprint 인스턴스 생성
# http://127.0.0.1:5000/
main_bp=Blueprint('root', __name__, url_prefix='/', template_folder='templates')

# 라우팅 기능 함수 정의
@main_bp.route('/', endpoint='hello') # endpoint를 안쓰면 함수이름이 노출. 플라스크에서만 이렇게 씀
def index():
    return render_template('index.html')