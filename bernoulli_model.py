# bernoulli_model.py

import numpy as np
from collections import defaultdict

class BernoulliModel:
    def __init__(self, num_classes, sorted_labels=None):
        """
        베르누이 모델 초기화
        :param num_classes: 클래스 개수
        :param sorted_labels: 정렬된 클래스 라벨 (기본값: None)
        """
        self.num_classes = num_classes
        self.sorted_labels = sorted_labels if sorted_labels is not None else list(range(num_classes))
        self.class_pixel_probs = defaultdict(lambda: None)
        self.class_counts = defaultdict(int)
        self.class_probs = None

    def train(self, data, labels):
        """
        데이터를 학습하여 클래스별 픽셀 발생 확률 계산
        :param data: 학습 데이터 (이미지 배열)
        :param labels: 해당 데이터의 레이블
        """
        img_shape = data.shape[1:]
        self.class_pixel_probs = defaultdict(lambda: np.zeros(img_shape))
        self.class_counts = defaultdict(int)
        
        for image, label in zip(data, labels):
            self.class_pixel_probs[label] += image
            self.class_counts[label] += 1

        # 클래스별 픽셀 확률 계산 (라플라스 스무딩 적용)
        for label in self.class_pixel_probs:
            self.class_pixel_probs[label] = (self.class_pixel_probs[label] + 1) / (self.class_counts[label] + 2)

        # 클래스별 사전 확률 계산
        total_count = sum(self.class_counts.values())
        self.class_probs = {label: self.class_counts[label] / total_count for label in self.class_counts}

    def predict(self, observed_image, smoothing=1e-10):
        """
        관찰된 이미지를 기반으로 후행 확률 계산
        :param observed_image: 관찰된 이미지
        :param smoothing: 스무딩 값
        :return: 각 클래스의 후행 확률
        """
        posterior_probs = np.zeros(self.num_classes)

        for idx, label in enumerate(self.sorted_labels):
            likelihood = (
                observed_image * np.log(self.class_pixel_probs[label] + smoothing) +
                (1 - observed_image) * np.log(1 - self.class_pixel_probs[label] + smoothing)
            )
            posterior_probs[idx] = likelihood.sum()

        # 지수화 및 정규화
        posterior_probs = np.exp(posterior_probs - np.max(posterior_probs))
        class_probs_list = [self.class_probs[label] for label in self.sorted_labels]
        posterior_probs *= class_probs_list
        posterior_probs /= posterior_probs.sum()

        return posterior_probs
