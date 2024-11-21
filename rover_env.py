# rover_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class RoverEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, image, bernoulli_model, true_label, render_mode=None):
        super().__init__()
        self.image = image
        self.model = bernoulli_model
        self.true_label = true_label
        self.height, self.width = image.shape[1], image.shape[2]
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.height, self.width), dtype=np.float32)
        self.render_mode = render_mode

        self.agent_pos = None
        self.visited = None
        self.step_count = 0
        self.max_steps = self.height * self.width
        self.num_classes = self.model.num_classes

        # 렌더링을 위한 matplotlib 설정
        self.fig, self.ax = None, None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [self.height // 2, self.width // 2]  # 중앙에서 시작
        self.visited = np.zeros_like(self.image)
        self.update_observation()
        self.step_count = 0
        observation = self.visited.copy()  # (3,64,64)
        info = {}
        if self.render_mode == 'human':
            self._init_render()
        return observation, info

    def update_observation(self):
        y, x = self.agent_pos
        self.visited[:, y, x] = 1  # 모든 채널에 방문 표시

    def step(self, action):
        y, x = self.agent_pos
        if action == 0 and y > 0:
            y -= 1  # 위로 이동
        elif action == 1 and y < self.height - 1:
            y += 1  # 아래로 이동
        elif action == 2 and x > 0:
            x -= 1  # 왼쪽으로 이동
        elif action == 3 and x < self.width - 1:
            x += 1  # 오른쪽으로 이동

        self.agent_pos = [y, x]
        self.update_observation()
        self.step_count += 1

        # 모델의 predict 메서드 사용
        observed = self.visited[0]  # 그레이스케일 채널 사용
        posterior_probs = self.model.predict(observed, smoothing=1e-2)

        # 관찰된 픽셀 비율 기반 가중치 계산
        num_pixels_observed = np.sum(observed)
        total_pixels = self.height * self.width
        observation_ratio = num_pixels_observed / total_pixels
        weight = observation_ratio  # 관찰된 픽셀 비율이 높을수록 우도에 더 큰 가중치

        # 가중 엔트로피 조정
        uniform_probs = np.ones(self.num_classes) / self.num_classes  # 균등 분포
        posterior_probs = weight * posterior_probs + (1 - weight) * uniform_probs

        predicted_label = self.model.sorted_labels[np.argmax(posterior_probs)]
        confidence = np.max(posterior_probs)

        # 확률 출력
        print(f"Posterior probabilities: {posterior_probs}")
        print(f"Predicted label: {predicted_label}, Confidence: {confidence}")

        # 보상 및 종료 조건 설정
        terminated = False
        truncated = False
        reward = -1  # 이동 비용
        if confidence > 0.7:
            terminated = True
            if predicted_label == self.true_label:
                reward += 100  # 정확한 예측 시 큰 보상
            else:
                reward -= 100  # 잘못된 예측 시 큰 패널티
        elif self.step_count >= self.max_steps:
            truncated = True  # 최대 스텝 수 도달 시 종료

        observation = self.visited.copy()  # (3,64,64)
        info = {'predicted_label': predicted_label, 'confidence': confidence}
        if self.render_mode == 'human':
            self.render()
        return observation, reward, terminated, truncated, info

    def _init_render(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title("Rover Exploration")
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.invert_yaxis()  # 이미지 좌표와 일치하도록 y축 반전
        # 그리드 라인 설정
        step_size_x = max(1, self.width // 10)
        step_size_y = max(1, self.height // 10)
        self.ax.set_xticks(range(0, self.width, step_size_x))
        self.ax.set_yticks(range(0, self.height, step_size_y))
        self.ax.grid(True)

    def render(self):
        if self.render_mode == 'human':
            if self.fig is None or self.ax is None:
                self._init_render()

            self.ax.clear()
            # 방문한 위치는 회색, 로버의 현재 위치는 빨간 점으로 표시
            self.ax.imshow(self.visited[0], cmap='gray', extent=(0, self.width, 0, self.height))
            self.ax.scatter(self.agent_pos[1] + 0.5, self.agent_pos[0] + 0.5, color='red', label='Rover', s=100)
            self.ax.set_title(f"Rover Exploration - Step {self.step_count}")
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.invert_yaxis()
            # 그리드 라인 다시 설정
            step_size_x = max(1, self.width // 10)
            step_size_y = max(1, self.height // 10)
            self.ax.set_xticks(range(0, self.width, step_size_x))
            self.ax.set_yticks(range(0, self.height, step_size_y))
            self.ax.grid(True)
            self.ax.legend(loc='upper right')
            plt.draw()
            plt.pause(0.5)  # 실시간 업데이트를 위한 대기 시간을 0.5초로 늘림
