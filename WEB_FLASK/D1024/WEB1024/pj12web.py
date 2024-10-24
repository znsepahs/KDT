from flask import Flask, render_template, request
import joblib
import pandas as pd

# Flask 애플리케이션 생성
app = Flask(__name__)

# 저장된 모델 불러오기
loaded_model = joblib.load('best_model.pkl')

# 홈 페이지 라우트 (index 페이지)
@app.route('/')
def home():
    return render_template('index.html')

# 예측 요청 처리 및 결과 페이지로 전송
@app.route('/predict', methods=['POST'])
def predict():
    # 사용자가 입력한 날짜 가져오기
    date_input = request.form['date']
    
    try:
        # 날짜를 타임스탬프로 변환
        date_timestamp = pd.to_datetime(date_input).timestamp()
        
        # 새로운 데이터 프레임 생성
        new_data = pd.DataFrame({'Date': [date_timestamp]})
        
        # 모델을 사용하여 예측
        predicted_price = loaded_model.predict(new_data)[0]
        
        # 결과 페이지로 이동
        return render_template('result.html', date=date_input, predicted_price=round(predicted_price, 2))
    
    except Exception as e:
        return f"<h2>Error: {e}</h2>"

# Flask 애플리케이션 실행
if __name__ == '__main__':
    app.run(debug=True)
