from rover_env import RoverEnv
from bernoulli_model import BernoulliModel
from data_loader import load_and_split_data
from sb3_contrib import RecurrentPPO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import time  # 시간 지연을 위해 추가

def main():
    # 데이터 로드
    train_images, test_images, train_labels, test_labels = load_and_split_data("./crack", img_size=(64, 64))
    label_names = ['0', '1', '3']
    label_to_index = {label_name: idx for idx, label_name in enumerate(label_names)}
    num_classes = len(label_names)
    sorted_labels = list(range(num_classes))  # [0, 1, 2]

    # 베르누이 모델 초기화 및 학습
    bernoulli_model = BernoulliModel(num_classes=num_classes, sorted_labels=sorted_labels)
    bernoulli_model.train(train_images, train_labels)

    # 테스트할 이미지 선택 (임의의 한 이미지)
    test_index = 0  # 원하는 테스트 이미지 인덱스로 변경 가능
    test_image = test_images[test_index]
    true_label = test_labels[test_index]

    # 환경 생성 (렌더 모드 활성화)
    env = RoverEnv(bernoulli_model, test_image, true_label, render_mode='human')

    # 환경 초기화
    obs, info = env.reset()

    # 테스트 이미지를 시각화
    display = np.full((env.height, env.width), 0, dtype=np.int32)  # 0: Not visited (Red)

    for channel in range(env.channels):
        visited_positions = np.where(env.visited[channel, :, :] == 255)
        for y, x in zip(visited_positions[0], visited_positions[1]):
            if test_image[channel, y, x] == 1:
                display[y, x] = 1  # Visited white
            elif test_image[channel, y, x] == 0:
                display[y, x] = 2  # Visited black

    # 색상 맵 정의
    cmap = ListedColormap(['red', 'white', 'black'])

    plt.figure(figsize=(6,6))
    plt.imshow(display, cmap=cmap, extent=(0, env.width, 0, env.height))
    plt.title("Initial Test Image")
    # 범례 추가
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Not Visited'),
        Patch(facecolor='white', edgecolor='black', label='Visited White'),
        Patch(facecolor='black', edgecolor='black', label='Visited Black')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.axis('off')
    plt.show()

    # 훈련된 에이전트 로드
    try:
        agent = RecurrentPPO.load("recurrent_ppo_rover_agent.zip", env=env)
    except FileNotFoundError:
        print("훈련된 에이전트 파일을 찾을 수 없습니다. 먼저 train_rl.py를 실행하여 에이전트를 훈련시키세요.")
        return

    # 에이전트 테스트
    obs, _ = env.reset()
    lstm_states = None  # 초기 LSTM 상태
    episode_starts = np.ones((1,), dtype=bool)

    done = False
    while not done:
        # 에이전트가 예측한 액션 선택
        action, lstm_states = agent.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=True
        )
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Action: {action}, Reward: {reward}, Info: {info}")

        # 움직임을 천천히 보기 위해 0.5초 대기
        time.sleep(0.5)

    # 렌더링 종료
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
