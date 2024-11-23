import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import RecurrentMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from data_loader import load_and_split_data
from bernoulli_model import BernoulliModel
from rover_env import RoverEnv
import gymnasium as gym

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)
        
        self.extractors = nn.ModuleDict()
        
        # 관찰 공간에서 각 부분의 특징 추출기 정의
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == 'image':
                n_input_channels = subspace.shape[0]
                self.extractors[key] = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                # CNN 출력 차원 계산
                with torch.no_grad():
                    sample_input = torch.zeros(1, *subspace.shape)
                    cnn_output_dim = self.extractors[key](sample_input).shape[1]
                total_concat_size += cnn_output_dim
            elif key == 'position':
                self.extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 64),
                    nn.ReLU(),
                )
                position_output_dim = 64
                total_concat_size += position_output_dim

        # 총 출력 차원 설정
        self._features_dim = total_concat_size

    def forward(self, observations):
        # 모델의 디바이스 가져오기
        device = next(self.parameters()).device
        # 관찰 값의 각 요소를 디바이스로 이동
        observations = {k: v.to(device) for k, v in observations.items()}
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == 'image':
                obs = obs.float() / 255.0  # 이미지 정규화
            else:
                obs = obs.float()
            encoded_tensor_list.append(extractor(obs))

        return torch.cat(encoded_tensor_list, dim=1)

def train_recurrent_ppo(env, num_timesteps=1000000):
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        lstm_hidden_size=256,
        net_arch=[256],
    )
    model = RecurrentPPO(
        policy=RecurrentMultiInputActorCriticPolicy,
        env=env,
        verbose=1,
        n_steps=128,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=4,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device='cuda',  # 모델을 GPU로 이동
    )
    model.learn(total_timesteps=num_timesteps)
    return model

if __name__ == "__main__":
    # 랜덤 시드 설정
    seed = 42
    set_random_seed(seed)

    # 데이터 로드 및 전처리
    train_images, test_images, train_labels, test_labels = load_and_split_data("./crack", img_size=(64, 64))
    label_names = ['0', '1', '3']
    label_to_index = {label_name: idx for idx, label_name in enumerate(label_names)}
    num_classes = len(label_names)
    sorted_labels = list(range(num_classes))  # [0, 1, 2]

    # 베르누이 모델 초기화 및 학습
    bernoulli_model = BernoulliModel(num_classes=num_classes, sorted_labels=sorted_labels)
    bernoulli_model.train(train_images, train_labels)

    # 환경 생성 함수 정의
    def make_env():
        def _init():
            idx = np.random.randint(len(train_images))
            env = RoverEnv(bernoulli_model, train_images[idx], train_labels[idx], render_mode=None)
            return env
        return _init

    # 환경 벡터화
    num_envs = 4
    envs = [make_env() for _ in range(num_envs)]
    env = DummyVecEnv(envs)

    # 에이전트 훈련
    agent = train_recurrent_ppo(env, num_timesteps=1000000)

    # 에이전트 저장
    agent.save("recurrent_ppo_rover_agent")
