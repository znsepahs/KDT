import os
from flask import Flask, render_template, request, send_file
import sys
import cv2
import numpy as np
import time
import tqdm

# Detectron2 및 프로젝트 종속성 import
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo
from adet.config import get_cfg

app = Flask(__name__)

# 업로드 및 출력 폴더 설정
UPLOAD_FOLDER = './demo/uploads/input'
OUTPUT_FOLDER = './demo/uploads/output'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
DEFAULT_WEIGHTS = "./pt_model/pretrain.pth"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def blur_text_regions(frame, predictions):
    """텍스트 영역 블러 처리 함수"""
    if "instances" not in predictions:
        return frame

    instances = predictions["instances"].to("cpu")
    
    if len(instances.polygons) == 0:
        return frame

    for polygon in instances.polygons:
        try:
            if hasattr(polygon, 'cpu'):
                polygon = polygon.cpu().numpy()
            
            poly = np.array(polygon).reshape((-1, 2)).astype(np.int32)
            
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            
            roi = cv2.bitwise_and(frame, frame, mask=mask)
            #blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
            blurred_roi = cv2.medianBlur(roi, 25)
            
            inv_mask = cv2.bitwise_not(mask)
            frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
            
            frame = cv2.add(frame_bg, blurred_roi)
        
        except Exception as e:
            print(f"Error processing polygon: {e}")
            continue
    
    return frame

def setup_cfg(config_file='./configs/DPText_DETR/Video.yaml', confidence_threshold=0.3, weights_path=DEFAULT_WEIGHTS):
    """설정 파일 로드 및 설정 함수"""
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHTS", weights_path])
    # 신뢰도 임계값 설정
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    
    cfg.freeze()
    return cfg

def process_video(input_path, output_path, confidence_threshold=0.3, weights_path=DEFAULT_WEIGHTS):
    """비디오 처리 함수"""
    # 로거 설정
    logger = setup_logger()
    logger.info(f"Processing video: {input_path}")

    # 설정 로드
    cfg = setup_cfg(confidence_threshold=confidence_threshold, weights_path=weights_path)
    demo = VisualizationDemo(cfg)

    # 비디오 입력 설정
    video = cv2.VideoCapture(input_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 비디오 출력 설정
    output_file = cv2.VideoWriter(
        filename=output_path,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )

    # 프레임 처리 및 블러 적용
    progress_bar = tqdm.tqdm(total=num_frames, unit='frames', desc='Processing')
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # 텍스트 검출
        predictions, _ = demo.run_on_image(frame)
        
        # 텍스트 영역 블러 처리
        blurred_frame = blur_text_regions(frame, predictions)
        
        # 결과 출력
        output_file.write(blurred_frame)
        
        progress_bar.update(1)

    # 자원 해제
    video.release()
    output_file.release()
    progress_bar.close()

    logger.info(f"Blurred video saved to {output_path}")

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        # 파일 확인
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        
        # 파일 이름 확인
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        # 파일 타입 확인
        if file and allowed_file(file.filename):
            # 고유한 파일명 생성
            
            input_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            output_filename = os.path.join(app.config['OUTPUT_FOLDER'], f'blurred_{file.filename}')
            
            # 파일 저장
            file.save(input_filename)
            
            # 텍스트 블러 처리 실행
            confidence = float(request.form.get('confidence', 0.3))
            
            weights_path=request.form.get('weights_path', DEFAULT_WEIGHTS)
            
            try:
                process_video(input_filename, output_filename, confidence, weights_path)
                return render_template('index.html', 
                                       output_video=f'uploads/output/blurred_{file.filename}', 
                                       original_filename=file.filename)
            
            except Exception as e:
                return render_template('index.html', error=f'Error processing video: {str(e)}')
    
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), 
                     download_name=f'blurred_{filename.split("_", 1)[1]}', 
                     as_attachment=True)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True)
