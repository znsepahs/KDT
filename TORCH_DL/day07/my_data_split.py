import os
import shutil
import random

def split_data(input_dir, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    # output 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    # train, val, test 폴더 생성
    for dir_path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 하위 폴더를 찾기 위한 루프
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)
            random.shuffle(files)  # 파일을 무작위로 섞음

            # 비율에 맞춰 파일 나누기
            total_files = len(files)
            train_files = files[:int(total_files * train_ratio)]
            val_files = files[int(total_files * train_ratio):int(total_files * (train_ratio + val_ratio))]
            test_files = files[int(total_files * (train_ratio + val_ratio)):]

            # 각각의 폴더로 파일을 이동
            for file in train_files:
                shutil.copy(os.path.join(subdir_path, file), os.path.join(train_dir, subdir + '_' + file))
            for file in val_files:
                shutil.copy(os.path.join(subdir_path, file), os.path.join(val_dir, subdir + '_' + file))
            for file in test_files:
                shutil.copy(os.path.join(subdir_path, file), os.path.join(test_dir, subdir + '_' + file))

    print(f"Data split completed. Train: {train_dir}, Validation: {val_dir}, Test: {test_dir}")

# 실행 예시
input_directory = r'C:\KDT\TORCH_DL\origin\data\path'  # 원본 데이터 폴더 경로
output_directory = r'C:\KDT\TORCH_DL\split\data\path'  # 결과를 저장할 폴더 경로

split_data(input_directory, output_directory)