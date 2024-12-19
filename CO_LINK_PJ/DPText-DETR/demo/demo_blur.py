# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import numpy as np
import tqdm

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg

# constants
WINDOW_NAME = "COCO detections"


def blur_text_regions(frame, predictions):
    """
    블러 처리를 위한 함수
    
    Args:
        frame (np.ndarray): 원본 프레임
        predictions (dict): 모델의 예측 결과
    
    Returns:
        np.ndarray: 텍스트 영역이 블러 처리된 프레임
    """
    if "instances" not in predictions:
        return frame

    # 검출된 인스턴스(텍스트)의 폴리곤 정보 가져오기
    instances = predictions["instances"].to("cpu")
    
    # 폴리곤이 없는 경우 예외 처리
    if len(instances.polygons) == 0:
        return frame

    for polygon in instances.polygons:
        try:
            # 폴리곤을 numpy 배열로 변환
            # 만약 폴리곤이 텐서인 경우 .numpy() 또는 .cpu().numpy() 사용
            if hasattr(polygon, 'cpu'):
                polygon = polygon.cpu().numpy()
            
            # 폴리곤 형태 검증 및 변환
            poly = np.array(polygon).reshape((-1, 2)).astype(np.int32)
            
            # 마스크 생성
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            
            # 관심 영역(ROI) 추출
            roi = cv2.bitwise_and(frame, frame, mask=mask)
            
            # 블러 처리 (가우시안, 평균값, 중간값 택 1)
            blurred_roi = cv2.GaussianBlur(roi, (31, 31), 0)
            #blurred_roi = cv2.blur(roi, (23, 23))
            #blurred_roi = cv2.medianBlur(roi, 23)

            # 블러 처리한 프레임에서 마스크 영역 제거
            blurred_roi = cv2.bitwise_and(blurred_roi, blurred_roi, mask=mask) 
            
            # 마스크의 반전된 영역
            inv_mask = cv2.bitwise_not(mask)
            
            # 원본 프레임에서 마스크 영역 제거
            frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
            
            # 블러 처리된 ROI와 원본 프레임 결합
            frame = cv2.add(frame_bg, blurred_roi)
        
        except Exception as e:
            print(f"Error processing polygon: {e}")
            continue
    
    return frame

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Text Blur Demo")
    parser.add_argument(
        "--config-file",
        default="./configs/DPText_DETR/Video.yaml",
        metavar="FILE",
        help="Path to config file",
    )
    parser.add_argument("--video-input", required=True, help="Path to input video file")
    parser.add_argument(
        "--output",
        default="./results/",
        help="Output video file path",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for text detection",
    )
    parser.add_argument(
        "--opts",
        default="MODEL.WEIGHTS ./pt_model/pretrain.pth",
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line 'KEY VALUE' pairs",
    )
    return parser

def main(args):
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    # 비디오 입력 설정
    video = cv2.VideoCapture(args.video_input)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 비디오 출력 설정
    output_fname = args.output
    output_file = cv2.VideoWriter(
        filename=output_fname,
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
        start_time = time.time()
        predictions, _ = demo.run_on_image(frame)
        
        # 텍스트 영역 블러 처리
        blurred_frame = blur_text_regions(frame, predictions)
        
        # 결과 출력
        output_file.write(blurred_frame)
        
        progress_bar.update(1)
        progress_bar.set_postfix({'processing_time': f'{time.time() - start_time:.2f}s'})

    # 자원 해제
    video.release()
    output_file.release()
    progress_bar.close()

    logger.info(f"Blurred video saved to {output_fname}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    main(args)
