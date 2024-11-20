# train_rl.py

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from rover_env import RoverEnv
from bernoulli_model import BernoulliModel
from data_loader import load_and_split_data
from stable_baselines3.common.vec_env import DummyVecEnv

def train_recurrent_ppo(env, num_timesteps=100000):
    model = RecurrentPPO(
        policy='MlpLstmPolicy',
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
        policy_kwargs=dict(
            net_arch=[128, 128],
            lstm_hidden_size=256,
        ),
    )
    model.learn(total_timesteps=num_timesteps)
    return model

if __name__ == "__main__":
    # Load data
    train_images, test_images, train_labels, test_labels = load_and_split_data("./crack", img_size=(448, 448))
    label_names = ['0', '1', '3']
    label_to_index = {label_name: idx for idx, label_name in enumerate(label_names)}
    num_classes = len(label_names)
    sorted_labels = list(range(num_classes))  # [0, 1, 2]

    # Initialize Bernoulli model
    bernoulli_model = BernoulliModel(num_classes=num_classes, sorted_labels=sorted_labels)
    bernoulli_model.train(train_images, train_labels)

    # Create environments for each test image
    def make_env(index):
        def _init():
            test_image = test_images[index]
            true_label = test_labels[index]
            env = RoverEnv(test_image, bernoulli_model, true_label)
            return env
        return _init

    envs = [make_env(i) for i in range(len(test_images))]
    env = DummyVecEnv(envs)

    # Train the agent
    agent = train_recurrent_ppo(env, num_timesteps=50000)
    agent.save("recurrent_ppo_rover_agent")
