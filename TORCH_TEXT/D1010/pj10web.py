from flask import Flask, request, render_template, redirect
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn 
from torchvision import models, transforms
from torchvision.models import VGG19_BN_Weights

import sys
sys.path.append(r'C:\WorkSpace\KDT\TORCH_IMAGE\MyModule\MyModule')
from KDTModule import *


app = Flask(__name__)

# 업로드할 이미지 저장 경로
UPLOAD_FOLDER = r'C:\WorkSpace\KDT\TORCH_IMAGE\D0927\WEBPAGE\WEBPAGE\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 업로드 폴더가 없으면 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')  # 'file' 키의 파일 가져오기

    # 파일이 존재하면 저장하고 텐서로 변환
    if file and file.filename:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)  # 파일 저장

            

        background_image = f'../static/{pred_answer}.jpg'

        return render_template('result.html', pred_answer=pred_answer, background_image=background_image)
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
