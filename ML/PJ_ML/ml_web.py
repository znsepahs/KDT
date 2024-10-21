from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Flask 앱 생성
app = Flask(__name__)

# 모델과 스케일러 불러오기
model = joblib.load('knn_model.pkl')
mmScaler = joblib.load('mmScaler.pkl')

# 기존 데이터셋의 컬럼
columns = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 
           'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race',
           'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma',
           'KidneyDisease', 'SkinCancer']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def ml_predict():
    # 입력 데이터를 HTML 폼으로부터 가져옴
    input_data_2d_17features = [request.form.getlist('feature')]
    new_data = pd.DataFrame(input_data_2d_17features, columns=columns)
    
    # 데이터 스케일링
    new_data_scaled = mmScaler.transform(new_data)
    
    # 예측 수행
    pred = model.predict(new_data_scaled)

    # 결과를 기반으로 출력 메시지 설정
    if pred.shape[0] == 1:
        return jsonify({"result": "심장병 발병 가능성이 높습니다. 근 시일 내에 내원하여 담당의의 진료를 받으십시오."})
    else:
        return jsonify({"result": "심장병 발병 가능성이 낮습니다."})

# 앱 실행
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
