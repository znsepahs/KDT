from flask import Flask, request, render_template, redirect
import os
from PIL import Image
import torch
import torch.nn as nn 
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from KDTModule import *
import pickle
import sys
import codecs
from konlpy.tag import Okt
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)


# index.html을 렌더링하는 경로
@app.route('/')
def index():
    return render_template('index.html')


# 각 범주에 맞는 HTML 페이지로 이동

@app.route('/hobby', methods=['GET'])
def hobby():
    return render_template('hobby.html')



# 입력된 데이터 처리하고 result.html로 넘기는 경로
# 취미/여가/여행
@app.route('/result_hobby', methods=['POST'])
def result_hobby():
    input_text = request.form['input_text'] # index.html에서 전송한 텍스트

    token = Okt().morphs(input_text)
    clean_text = remove_stopwords(token, 'stopword.txt')
    clean_text = remove_punctuation(clean_text)
    join_clean_text = ' '.join(clean_text)

    inputDF = pd.DataFrame([join_clean_text], columns=['clean_text'])

    best_model = LSTMModel(input_size = 5000, output_size = 8, hidden_list = [100, 30],
                        act_func=F.relu, model_type='multiclass', num_layers=1)
    # 본인 pth 경로로 바꾸기
    pth_PATH = r'C:\Users\LG\OneDrive\바탕 화면\WEBPAGE4\WEBPAGE4/text_classification_model.pth'
    best_model.load_state_dict(torch.load(pth_PATH, weights_only=True))

    best_model.eval()
    # 본인 pkl 경로로 바꾸기
    pkl_PATH = r'C:\Users\LG\OneDrive\바탕 화면\WEBPAGE4\WEBPAGE4/tfid_vectorizer_5000.pkl'
    loaded_vectorizer = joblib.load(pkl_PATH)
    input_vector = loaded_vectorizer.transform(inputDF['clean_text'].values)
    input_tensor_vector = torch.FloatTensor(input_vector.toarray())
    with torch.no_grad(): # 불필요한 그레디언트 계산 끄기 (메모리 사용량 줄이기) *켜도 eval 결과에 영향은 없음*
        input_logit = best_model(input_tensor_vector.unsqueeze(1))
        pred = torch.argmax(input_logit, dim=1).item()

    section_dict = {0 : '국내 여행', 1 : '게임', 2 : '취미', 3 : '해외 여행',
                    4 : '사진', 5 : '맛집', 6 : '스포츠', 7 : '자동차'}

    pred_section = section_dict[pred]

    return render_template('result_hobby.html', pred_section=pred_section)

if __name__ == '__main__':
    app.run(debug=True, port=5004)