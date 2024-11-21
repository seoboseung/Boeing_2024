# main.py 또는 test_scripts/test_model.py

import matplotlib.pyplot as plt
import numpy as np
from bernoulli_model import BernoulliModel
from data_loader import load_and_split_data

def test_expanding_partial_images_with_weighted_entropy(model, test_image, sorted_labels, steps=5, smoothing=1e-2, alpha=0.5):
    """
    다양한 크기의 부분 이미지를 모델에 입력하여 확률 계산 및 시각화 (가중 엔트로피 조정 포함).
    :param model: 학습된 BernoulliModel 인스턴스
    :param test_image: 테스트 이미지 (3, H, W)
    :param sorted_labels: 정렬된 클래스 라벨 리스트
    :param steps: 관찰 스텝 수
    :param smoothing: 스무딩 상수
    :param alpha: 포스터리어 확률과 균등 분포의 혼합 비율 (0 < alpha < 1)
    """
    height, width = test_image.shape[1], test_image.shape[2]
    center_y, center_x = height // 2, width // 2  # 중앙 기준

    # 테스트 이미지를 이진화 (그레이스케일 채널 사용)
    binary_image = test_image[0]  # 그레이스케일 채널 가정, (H, W)

    num_classes = len(sorted_labels)

    for step in range(steps):
        # 관찰 영역 크기 계산
        current_window_size = (step + 1) * (max(height, width) // steps)
        half_window = current_window_size // 2

        y_start = max(0, center_y - half_window)
        y_end = min(height, center_y + half_window)
        x_start = max(0, center_x - half_window)
        x_end = min(width, center_x + half_window)

        # 부분 이미지 생성
        observed = np.zeros_like(binary_image)
        observed[y_start:y_end, x_start:x_end] = binary_image[y_start:y_end, x_start:x_end]

        # 모델의 predict 메서드를 사용하여 후행 확률 계산
        posterior_probs = model.predict(observed, smoothing=smoothing)

        # 가중 엔트로피 조정 (alpha 사용)
        uniform_probs = np.ones(num_classes) / num_classes  # 균등 분포
        posterior_probs = alpha * posterior_probs + (1 - alpha) * uniform_probs

        # 결과 시각화
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(observed, cmap="gray")
        plt.title(f"Step {step + 1}: Observed Region")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.bar(range(num_classes), posterior_probs, color='skyblue')
        plt.title("Probability Distribution")
        plt.xlabel("Labels")
        plt.ylabel("Probability")
        plt.xticks(range(num_classes), labels=sorted_labels)
        plt.ylim(0, 1)
        plt.show()
        print(f"Step {step + 1} Posterior Probabilities: {posterior_probs}")

if __name__ == "__main__":
    # 데이터 로드 및 모델 학습
    train_images, test_images, train_labels, test_labels = load_and_split_data("./crack", img_size=(64, 64))
    sorted_labels = ['0', '1', '2']  # 실제 라벨에 맞게 설정
    bernoulli_model = BernoulliModel(num_classes=3, sorted_labels=sorted_labels)
    bernoulli_model.train(train_images, train_labels, alpha=1.0)

    # 테스트할 이미지 선택
    test_index = 33  # 원하는 테스트 이미지 인덱스로 변경
    test_image = test_images[test_index]
    test_label = test_labels[test_index]
    print(f"True Label: {test_label}")

    # 테스트 함수 호출
    test_expanding_partial_images_with_weighted_entropy(
        bernoulli_model, test_image, sorted_labels, steps=5, smoothing=1e-2, alpha=0.3  # alpha 값을 조정
    )
