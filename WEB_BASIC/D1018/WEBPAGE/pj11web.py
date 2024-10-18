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
UPLOAD_FOLDER_1 = r'C:\WorkSpace\KDT\TORCH_DL\day07\WEBPAGE\uploads'
UPLOAD_FOLDER_2 = r'C:\WorkSpace\KDT\TORCH_IMAGE\D0927\WEBPAGE\WEBPAGE\uploads'
app.config['UPLOAD_FOLDER_1'] = UPLOAD_FOLDER_1
app.config['UPLOAD_FOLDER_2'] = UPLOAD_FOLDER_2

# 업로드 폴더가 없으면 생성
os.makedirs(UPLOAD_FOLDER_1, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_2, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')  # 'file' 키의 파일 가져오기
    model_type = request.form.get('model_type')  # 사용자가 선택한 모델 타입 가져오기

    if file and file.filename:
        # 모델 타입에 따라 다른 폴더에 저장
        if model_type == 'model1':
            file_path = os.path.join(app.config['UPLOAD_FOLDER_1'], file.filename)
            name_dict = {0: 'cane', 1: 'cavallo', 2: 'elefante', 3: 'farfalla', 4: 'gallina', 5: 'gatto', 6: 'mucca', 7: 'pecora', 8: 'ragno', 9: 'scoiattolo'}
            pth_PATH = r'C:\WorkSpace\LocalData\pj_08_DL\miniP\vgg19_bn_model\best_model_epoch_1.pth'
            num_classes = 10
            
        elif model_type == 'model2':
            file_path = os.path.join(app.config['UPLOAD_FOLDER_2'], file.filename)
            name_dict = {0: '짱구', 1: '철수', 2: '훈이', 3: '맹구', 4: '유리'}
            pth_PATH = r'C:\WorkSpace\LocalData\pj_09_DL\model\best_model_epoch_5.pth'
            num_classes = 5

        file.save(file_path)  # 파일 저장

        # 모델 설정
        best_model = models.vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)  # 전이학습 모델 불러오기
        best_model.classifier = nn.Linear(in_features=25088, out_features=num_classes)  # 전결합층 입력 출력 변경
        best_model.load_state_dict(torch.load(pth_PATH, map_location=torch.device('cpu')))  # 모델에 가중치 설정

        # 이미지 변환
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 정규화 값
        ])

        image = Image.open(file_path).convert('RGB')
        tensor_image = transform(image)
        input_tensor = tensor_image.unsqueeze(0)

        # 모델 예측
        best_model.eval()
        pred = torch.argmax(best_model(input_tensor), dim=1).item()
        pred_answer = name_dict[pred]

        background_image = f'../static/{pred_answer}.jpg'

        return render_template('result.html', pred_answer=pred_answer, background_image=background_image)

    return redirect(request.url)


if __name__ == '__main__':
    # IP와 포트를 원하는 값으로 설정 (예: 0.0.0.0:8080)
    app.run(host='127.0.0.1', port=5002)
