# main.py

from rover_env import RoverEnv
from bernoulli_model import BernoulliModel
from data_loader import load_and_split_data
from sb3_contrib import RecurrentPPO
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 데이터 로드 및 모델 학습
    train_images, test_images, train_labels, test_labels = load_and_split_data("./crack", img_size=(64, 64))
    label_names = ['0', '1', '3']
    label_to_index = {label_name: idx for idx, label_name in enumerate(label_names)}
    num_classes = len(label_names)
    sorted_labels = list(range(num_classes))  # [0, 1, 2]

    # Bernoulli 모델 초기화 및 학습
    bernoulli_model = BernoulliModel(num_classes=num_classes, sorted_labels=sorted_labels)
    bernoulli_model.train(train_images, train_labels)

    # 테스트 이미지 선택
    test_index = 0  # 테스트할 이미지 인덱스 선택
    test_image = test_images[test_index]
    true_label = test_labels[test_index]

    # 환경 생성 (렌더 모드 활성화)
    env = RoverEnv(test_image, bernoulli_model, true_label, render_mode='human')

    # 훈련된 에이전트 로드
    agent = RecurrentPPO.load("recurrent_ppo_rover_agent")

    # 에이전트 테스트
    obs, _ = env.reset()
    lstm_states = None  # 초기 LSTM 상태
    episode_starts = np.ones((1,), dtype=bool)

    done = False
    while not done:
        action, lstm_states = agent.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_starts = np.array([done])

        print(f"Action: {action}, Reward: {reward}, Info: {info}")

    # 렌더링 종료
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
