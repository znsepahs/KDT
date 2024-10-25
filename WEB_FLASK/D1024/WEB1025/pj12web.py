from flask import Flask, render_template, request
import joblib
import pandas as pd

# Flask 애플리케이션 생성
app = Flask(__name__)

# 저장된 모델 불러오기
loaded_model = joblib.load('../../../../LocalData/pj_12_flask/model/best_model_02.pkl')

# 홈 페이지 (입력 폼)
@app.route('/')
def home():
    return render_template('index.html')

# 예측 요청 처리
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 사용자가 입력한 값 가져오기
        open_price = float(request.form['open'])
        high_price = float(request.form['high'])
        low_price = float(request.form['low'])

        # 입력값을 기반으로 새로운 데이터 생성
        new_data = pd.DataFrame({
            'Open': [open_price],
            'High': [high_price],
            'Low': [low_price]
        })

        # 예측 수행
        predicted_price = loaded_model.predict(new_data)[0]

        # 결과 페이지로 전달
        return render_template('result.html', predicted_price=round(predicted_price, 2))

    except Exception as e:
        return f"<h2>Error: {e}</h2>"

# Flask 애플리케이션 실행
if __name__ == '__main__':
    app.run(debug=True)
