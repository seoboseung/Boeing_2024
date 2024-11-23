# data_loader.py

import os
import numpy as np
from PIL import Image, ImageOps  # Resampling 제거
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def resize_with_padding(img, target_size=(64, 64)):
    """
    비율을 유지하며 이미지를 리사이즈하고, 부족한 부분을 패딩으로 채웁니다.
    
    :param img: PIL.Image 객체
    :param target_size: 목표 크기 (width, height)
    :return: 리사이즈 및 패딩된 PIL.Image 객체
    """
    img = img.copy()  # 원본 이미지 유지
    img.thumbnail(target_size, Image.Resampling.LANCZOS)  # Image.Resampling.LANCZOS 사용
    delta_width = target_size[0] - img.size[0]
    delta_height = target_size[1] - img.size[1]
    padding = (delta_width//2, delta_height//2, delta_width - (delta_width//2), delta_height - (delta_height//2))
    new_img = ImageOps.expand(img, padding, fill=0)  # 검은색 패딩
    return new_img

def load_crack_data_by_folder(base_dir, img_size=(64, 64), threshold=0.5):
    """
    지정된 폴더 구조에서 이미지를 로드하고 전처리합니다.
    
    :param base_dir: 데이터셋의 기본 디렉토리 경로
    :param img_size: 이미지 크기 (가로, 세로)
    :param threshold: 이진화 임계값
    :return: 이미지 배열과 라벨 배열
    """
    images = []
    labels = []

    # 데이터셋에 존재하는 라벨 정의
    label_names = ['0', '1', '3']
    label_to_index = {label_name: idx for idx, label_name in enumerate(label_names)}

    for label_name in label_names:
        label_idx = label_to_index[label_name]
        label_path = os.path.join(base_dir, label_name)
        if not os.path.isdir(label_path):
            continue  # 폴더가 없으면 건너뜀

        for filename in os.listdir(label_path):
            if filename.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(label_path, filename)
                try:
                    # 이미지 로드 및 전처리
                    img = Image.open(img_path).convert("L")
                    img = resize_with_padding(img, target_size=img_size)
                    img = np.array(img) / 255.0  # (H, W)
                    binary_img = (img > threshold).astype(np.float32)  # (H, W)
                    binary_img = np.expand_dims(binary_img, axis=0)  # (1,H, W)

                    images.append(binary_img)
                    labels.append(label_idx)  # 매핑된 라벨 인덱스 사용

                    # 첫 몇 개의 이미지를 시각적으로 확인
                    if len(images) <= 5:
                        plt.imshow(binary_img[0], cmap='gray')  # 채널 차원 제거
                        plt.title(f"Label: {label_name}")
                        plt.show()

                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels)

def load_and_split_data(base_dir, img_size=(64, 64), test_size=0.2, random_state=42):
    """
    데이터를 로드하고 훈련/테스트 세트로 분할합니다.
    
    :param base_dir: 데이터셋의 기본 디렉토리 경로
    :param img_size: 이미지 크기 (가로, 세로)
    :param test_size: 테스트 세트의 비율
    :param random_state: 랜덤 시드
    :return: 훈련 이미지, 테스트 이미지, 훈련 라벨, 테스트 라벨
    """
    images, labels = load_crack_data_by_folder(base_dir, img_size=img_size)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return train_images, test_images, train_labels, test_labels
