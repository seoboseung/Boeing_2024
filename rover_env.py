# rover_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import time

class RoverEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, image, bernoulli_model, true_label, render_mode=None):
        super().__init__()
        self.image = image
        self.model = bernoulli_model
        self.true_label = true_label
        self.height, self.width = image.shape
        self.action_space = spaces.Discrete(4)  # 상하좌우 이동
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, self.height, self.width), dtype=np.float32)
        self.render_mode = render_mode

        self.agent_pos = None
        self.visited = None
        self.step_count = 0
        self.max_steps = self.height * self.width

        # 렌더링을 위한 플롯 초기화
        self.fig, self.ax = None, None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [self.height // 2, self.width // 2]  # 중앙에서 시작
        self.visited = np.zeros_like(self.image)
        self.update_observation()
        self.step_count = 0
        observation = self.visited[np.newaxis, :, :]  # 채널 차원 추가
        info = {}
        if self.render_mode == 'human':
            self._init_render()
        return observation, info

    def update_observation(self):
        y, x = self.agent_pos
        self.visited[y, x] = self.image[y, x]

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

        # 관찰된 부분으로 예측
        posterior_probs = self.model.predict(self.visited)
        predicted_label = self.model.sorted_labels[np.argmax(posterior_probs)]
        confidence = np.max(posterior_probs)

        # 종료 조건 및 보상 설정
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

        observation = self.visited[np.newaxis, :, :]  # 채널 차원 추가
        info = {'predicted_label': predicted_label, 'confidence': confidence}
        if self.render_mode == 'human':
            self.render()
        return observation, reward, terminated, truncated, info

    def _init_render(self):
        """ 렌더링 초기화 """
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title("Rover Exploration")
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.invert_yaxis()  # 이미지 좌표와 일치하도록 y축 반전
        self.ax.set_xticks(range(self.width))
        self.ax.set_yticks(range(self.height))
        self.ax.grid(True)

    def render(self):
        """ 로버의 움직임 시각화 """
        if self.render_mode == 'human':
            if self.fig is None or self.ax is None:
                self._init_render()

            self.ax.clear()
            self.ax.imshow(self.visited, cmap='gray', extent=(0, self.width, 0, self.height))
            self.ax.scatter(self.agent_pos[1] + 0.5, self.agent_pos[0] + 0.5, color='red', label='Rover', s=100)
            self.ax.set_title(f"Rover Exploration - Step {self.step_count}")
            self.ax.set_xticks(range(self.width))
            self.ax.set_yticks(range(self.height))
            self.ax.grid(True)
            plt.draw()
            plt.pause(0.1)  # 실시간 업데이트를 위한 대기
