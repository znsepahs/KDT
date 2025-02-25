from flask import Flask, request, render_template, redirect
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn 
from torchvision import models, transforms
from torchvision.models import VGG19_BN_Weights

import sys
sys.path.append(r'C:\WorkSpace\KDT\MyModule')
from DLModule import *


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

        name_dict = {0 : '짱구', 1 : '철수', 2: '훈이' , 3 : '맹구', 4 : '유리'}

        best_model = models.vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1) # 전이학습 모델 불러오기
        best_model.classifier = nn.Linear(in_features=25088, out_features=5) # 전결합층 입력 출력 변경
        pth_PATH = r'C:\WorkSpace\LocalData\pj_09_DL\model\best_model_epoch_5.pth'
        best_model.load_state_dict(torch.load(pth_PATH, map_location=torch.device('cpu'))) # 모델에 가중치 설정

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet 정규화 값
        ])

        image = Image.open(file_path).convert('RGB')
        tensor_image = transform(image)
        input_tensor = tensor_image.unsqueeze(0)
        best_model.eval()
        pred = torch.argmax(best_model(input_tensor), dim=1).item()
        pred_answer = name_dict[pred]     

        background_image = f'../static/{pred_answer}.jpg'

        return render_template('result.html', pred_answer=pred_answer, background_image=background_image)
    
    return redirect(request.url)

if __name__ == '__main__':
    # IP와 포트를 원하는 값으로 설정 (예: 0.0.0.0:8080)
    app.run(host='127.0.0.1', port=5003)