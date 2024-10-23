from flask import Blueprint, render_template
from DBWEB.models.models import Question

mainBP=Blueprint('question', __name__, url_prefix='/', template_folder='templates')

@mainBP.route('/list/')
def _list():
    question_list=Question.query.order_by(Question.create_date.desc())
    return render_template('question_list.html',
                           question_list=question_list)

@mainBP.route('/qdetail/<int:question_id>/')
def detail(question_id):
    question=Question.query.get_or_404(question_id)
    return render_template('question_detail.html', question=question)