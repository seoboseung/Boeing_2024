import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

class RoverEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, bernoulli_model, image, label, render_mode=None):
        super().__init__()
        self.model = bernoulli_model
        self.image = image  # (C, H, W) 형상
        self.true_label = label
        self.channels, self.height, self.width = self.image.shape
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(self.channels + 1, self.height, self.width), dtype=np.uint8),
            'position': spaces.Box(low=0, high=max(self.height, self.width), shape=(2,), dtype=np.int32),
            'posterior_probs': spaces.Box(low=0.0, high=1.0, shape=(self.num_classes,), dtype=np.float32),
        })

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
        self.agent_pos = [self.height // 2, self.width // 2]
        # 채널을 하나 추가하여 방문 정보를 저장
        self.visited = np.zeros((self.channels + 1, self.height, self.width), dtype=np.uint8)
        # 원본 이미지 채널 복사
        self.visited[:self.channels] = self.image
        self.update_observation()
        self.step_count = 0
        observation = {
            'image': self.visited.copy(),
            'position': np.array(self.agent_pos, dtype=np.int32),
            'posterior_probs': np.ones(self.num_classes, dtype=np.float32) / self.num_classes,  # 초기에는 균등 분포
        }
        info = {}
        if self.render_mode == 'human':
            self._init_render()
        return observation, info

    def update_observation(self):
        y, x = self.agent_pos
        # 방문 정보 채널에 방문 표시
        self.visited[self.channels, y, x] = 255

    def step(self, action):
        y, x = self.agent_pos
        new_y, new_x = y, x  # 잠정적인 새로운 위치

        collision = False  # 충돌 여부 초기화

        # 이동 시도
        if action == 0 and y > 0:
            new_y -= 1  # 위로 이동
        elif action == 1 and y < self.height - 1:
            new_y += 1  # 아래로 이동
        elif action == 2 and x > 0:
            new_x -= 1  # 왼쪽으로 이동
        elif action == 3 and x < self.width - 1:
            new_x += 1  # 오른쪽으로 이동
        else:
            collision = True  # 벽에 부딪힘

        # 충돌 여부에 따른 처리
        if not collision:
            # 에이전트 위치 업데이트
            self.agent_pos = [new_y, new_x]
            self.update_observation()
            reward = -1  # 이동 비용
        else:
            # 이동 불가 시 보상 및 종료 처리
            reward = -10  # 벽에 부딪힘에 대한 패널티
            terminated = False
            truncated = False
            info = {'collision': True}

            # 관찰 값 반환
            observation = {
                'image': self.visited.copy(),
                'position': np.array(self.agent_pos, dtype=np.int32)
            }
            print(f"벽에 부딪혔습니다. 액션: {action}. 이동 불가. 패널티: -10")
            return observation, reward, terminated, truncated, info

        self.step_count += 1

        
        # 모델의 predict 메서드 사용
        observed = self.visited.mean(axis=0)  # 모든 채널 평균
        posterior_probs = self.model.predict(observed, smoothing=1e-2)

        # 관찰된 픽셀 비율 기반 가중치 계산
        num_pixels_observed = np.sum(self.visited > 0)
        total_pixels = self.height * self.width
        observation_ratio = num_pixels_observed / total_pixels
        weight = observation_ratio  # 관찰된 픽셀 비율이 높을수록 우도에 더 큰 가중치

        # 가중 엔트로피 조정
        uniform_probs = np.ones(self.num_classes) / self.num_classes  # 균등 분포
        posterior_probs = weight * posterior_probs + (1 - weight) * uniform_probs

        predicted_label = self.model.sorted_labels[np.argmax(posterior_probs)]
        confidence = np.max(posterior_probs)

        # 시간에 따른 음의 보상
        time_penalty = -0.01 * self.step_count  # 매 스텝마다 -0.01 보상
        reward += time_penalty  # 이동 비용과 시간 기반 음의 보상 합산

        # 흰색 픽셀 탐지 시 추가 보상
        current_pixel = self.image[:, new_y, new_x]  # 현재 위치의 픽셀 값
        if np.any(current_pixel > 128):  # 흰색 픽셀로 간주
            reward += 5  # 흰색 픽셀 보상
            print(f"흰색 픽셀을 발견했습니다. 추가 보상: +5")

        # 보상 및 종료 조건 설정
        terminated = False
        truncated = False

        if confidence > 0.7:
            terminated = True
            if predicted_label == self.true_label:
                reward += 100  # 정확한 예측 시 보상
                print(f"정확한 예측을 완료했습니다. 보상: +100")
            else:
                reward -= 50  # 잘못된 예측 시 패널티
                print(f"잘못된 예측을 했습니다. 패널티: -50")
        elif self.step_count >= self.max_steps:
            truncated = True  # 최대 스텝 수 도달 시 종료
            print("최대 스텝 수에 도달했습니다. 에피소드 종료.")

        # 확률 및 보상 출력
        print(f"Posterior probabilities: {posterior_probs}")
        print(f"Predicted label: {predicted_label}, Confidence: {confidence}")
        print(f"Time Penalty: {time_penalty}, Reward: {reward}")

        # info 딕셔너리에 현재 상태의 확률 추가
        info = {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'posterior_probs': posterior_probs,
            'collision': collision
        }

        # 관찰 값 반환
        observation = {
            'image': self.visited.copy(),
            'position': np.array(self.agent_pos, dtype=np.int32)
        }

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

            # 색상 맵 정의
            # 0: 방문하지 않음 (빨간색)
            # 1: 흰색 픽셀 방문 (흰색)
            # 2: 검은색 픽셀 방문 (검은색)
            display = np.full((self.height, self.width), 0, dtype=np.int32)  # 0: Not visited (Red)

            # 방문한 픽셀 설정
            for channel in range(3):
                visited_positions = np.where(self.visited[:, :, channel] == 1)
                for y, x in zip(visited_positions[0], visited_positions[1]):
                    if self.image[y, x, channel] == 1:
                        display[y, x] = 1  # Visited white
                    elif self.image[y, x, channel] == 0:
                        display[y, x] = 2  # Visited black

            # 색상 맵 생성
            cmap = ListedColormap(['red', 'white', 'black'])

            # 방문한 위치와 로버의 현재 위치를 시각화
            self.ax.imshow(display, cmap=cmap, extent=(0, self.width, 0, self.height))
            self.ax.scatter(self.agent_pos[1] + 0.5, self.agent_pos[0] + 0.5, color='yellow', label='Rover', s=100)
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

            # 범례 추가
            legend_elements = [
                Patch(facecolor='red', edgecolor='black', label='Not Visited'),
                Patch(facecolor='white', edgecolor='black', label='Visited White'),
                Patch(facecolor='black', edgecolor='black', label='Visited Black'),
                Patch(facecolor='yellow', edgecolor='black', label='Rover')
            ]
            self.ax.legend(handles=legend_elements, loc='upper right')

            plt.draw()
            plt.pause(0.5)  # 실시간 업데이트를 위한 대기 시간을 0.5초로 설정
