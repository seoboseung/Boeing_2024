# train_rl.py

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from rover_env import RoverEnv
from bernoulli_model import BernoulliModel
from data_loader import load_and_split_data
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env  # 환경 체크 도구 추가

def train_recurrent_ppo(env, num_timesteps=100000):
    """
    Recurrent PPO 에이전트를 훈련시킵니다.
    
    :param env: 강화학습 환경
    :param num_timesteps: 총 학습 타임스텝
    :return: 훈련된 에이전트
    """
    model = RecurrentPPO(
        policy='CnnLstmPolicy',  # 이미지 데이터를 처리하기 위한 CNN 기반 LSTM 정책 사용
        env=env,
        verbose=1,
        n_steps=128,
        batch_size=128,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=2.5e-4,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,  # policy_kwargs에서 use_sde 제거
        policy_kwargs=dict(
            lstm_hidden_size=256,
            normalize_images=False,  # 이미지 정규화를 비활성화
        ),
    )
    model.learn(total_timesteps=num_timesteps)
    return model

if __name__ == "__main__":
    # 데이터 로드
    train_images, test_images, train_labels, test_labels = load_and_split_data("./crack", img_size=(64, 64))
    label_names = ['0', '1', '3']
    label_to_index = {label_name: idx for idx, label_name in enumerate(label_names)}
    num_classes = len(label_names)
    sorted_labels = list(range(num_classes))  # [0, 1, 2]

    # 베르누이 모델 초기화 및 학습
    bernoulli_model = BernoulliModel(num_classes=num_classes, sorted_labels=sorted_labels)
    bernoulli_model.train(train_images, train_labels)

    # 환경 체크 (옵션)
    if len(train_images) > 0:
        sample_env = RoverEnv(train_images[0], bernoulli_model, train_labels[0])
        check_env(sample_env, warn=True)

    # 여러 훈련 이미지를 위한 환경 생성
    def make_env(index):
        def _init():
            train_image = train_images[index]
            true_label = train_labels[index]
            env = RoverEnv(train_image, bernoulli_model, true_label)
            return env
        return _init

    # 환경 벡터화
    envs = [make_env(i) for i in range(len(train_images))]
    env = DummyVecEnv(envs)

    # 에이전트 훈련
    agent = train_recurrent_ppo(env, num_timesteps=50000)

    # 에이전트 저장
    agent.save("recurrent_ppo_rover_agent")
