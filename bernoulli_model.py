import numpy as np
from collections import defaultdict

class BernoulliModel:
    def __init__(self, num_classes, sorted_labels=None):
        """
        베르누이 모델 초기화
        
        :param num_classes: 클래스 개수
        :param sorted_labels: 정렬된 클래스 라벨 리스트
        """
        self.num_classes = num_classes
        self.sorted_labels = sorted_labels if sorted_labels is not None else list(range(num_classes))
        self.class_pixel_probs = defaultdict(lambda: None)
        self.class_counts = defaultdict(int)
        self.class_probs = None

    def train(self, data, labels, alpha=1.0):
        img_shape = data.shape[1:]  # (C, H, W)
        self.class_pixel_probs = defaultdict(lambda: np.zeros(img_shape[1:]))  # (H, W)
        self.class_counts = defaultdict(int)
        
        for image, label in zip(data, labels):
            grayscale_image = image[0, :, :]  # 첫 번째 채널 사용 (이미 (C, H, W) 형상)
            self.class_pixel_probs[label] += grayscale_image
            self.class_counts[label] += 1

        # 라플라스 스무딩 적용 및 클래스 픽셀 확률 클램핑
        for label in self.class_pixel_probs:
            # Apply Laplace smoothing with alpha
            self.class_pixel_probs[label] = (self.class_pixel_probs[label] + alpha) / (self.class_counts[label] + 2 * alpha)
            # Clamp probabilities to avoid extremes
            self.class_pixel_probs[label] = np.clip(self.class_pixel_probs[label], 0.05, 0.95)

        # 클래스별 사전 확률 계산
        total_count = sum(self.class_counts.values())
        self.class_probs = {label: self.class_counts[label] / total_count for label in self.class_counts}

        # 학습 결과 출력 (디버깅용)
        print(f"Trained class probabilities: {self.class_probs}")
        for label in self.class_pixel_probs:
            print(f"Class {label} pixel probabilities mean: {self.class_pixel_probs[label].mean()}")

    def predict(self, observed_image, smoothing=1e-10):
        """
        관찰된 이미지를 기반으로 후행 확률을 계산합니다.
        
        :param observed_image: 관찰된 이미지 (이진화된 이미지, shape=(C, H, W) 또는 (H, W))
        :param smoothing: 로그 계산 시 0을 피하기 위한 스무딩 값
        :return: 각 클래스의 후행 확률 (shape=(num_classes,))
        """
        # observed_image의 형상이 (C, H, W) 또는 (H, W)일 수 있으므로 처리
        if observed_image.ndim == 3:
            observed_image = observed_image[0, :, :]
        elif observed_image.ndim == 2:
            pass  # 이미 (H, W) 형상이므로 그대로 사용
        else:
            raise ValueError(f"Invalid shape for observed_image: {observed_image.shape}")

        observed_image = observed_image / 255.0  # (H, W)

        log_likelihoods = np.zeros(self.num_classes)

        for idx, label in enumerate(self.sorted_labels):
            prob = self.class_pixel_probs[label]  # (H, W)
            # Compute log-likelihood
            log_prob = observed_image * np.log(prob + smoothing) + (1 - observed_image) * np.log(1 - prob + smoothing)
            log_likelihoods[idx] = log_prob.sum()

        # To prevent numerical instability, subtract the max log-likelihood
        max_log_likelihood = np.max(log_likelihoods)
        log_likelihoods -= max_log_likelihood
        # Exponentiate log-likelihoods
        likelihoods = np.exp(log_likelihoods)
        # Multiply by prior class probabilities
        class_probs = np.array([self.class_probs[label] for label in self.sorted_labels])
        posterior_probs = likelihoods * class_probs
        # Normalize to sum to 1
        posterior_probs /= posterior_probs.sum()

        return posterior_probs
