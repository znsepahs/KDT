#--------------------------------------------------------------------------
# Flask Framework에서 모듈단위 URL 처리 파일
# - 파일명 : main_view.py
#--------------------------------------------------------------------------
# 모듈로딩
from flask import Blueprint, render_template, url_for
from DBWEB.models.models import Question
from werkzeug.utils import redirect

# Blueprint 인스턴스 생성
mainBP=Blueprint('MAIN', import_name=__name__, url_prefix='/', template_folder='templates')

# http://localhost:5000/ URL 처리 라우팅 함수 정의
@mainBP.route("/")
def index():
    return redirect(url_for('question._list'))

# http://localhost:5000/ URL 처리 라우팅 함수 정의
@mainBP.route("/qlist")
def printList():
    # DB에서 조회한 결과를 HTML 파일로 전달
    q_list=Question.query.all()
    return render_template('question_list.html', question_list=q_list)

# http://localhost:5000/ URL 처리 라우팅 함수 정의
@mainBP.route("/qdetail/<int:question_id>/")
def questionItem(question_id):
    # DB에서 조회한 1개의 question 인스턴스를 전달
    question=Question.query.get_or_404(question_id)
    return render_template('question_detail.html', question=question)