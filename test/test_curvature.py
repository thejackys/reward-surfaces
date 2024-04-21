import gym
import numpy as np
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.ddpg import DDPG
from stable_baselines3.td3 import TD3
from stable_baselines3.sac import SAC
# from stable_baselines3.her import HER
import tempfile
import gym  # open ai gym
from stable_baselines3.common.envs import BitFlippingEnv
from reward_surfaces.agents import SB3OnPolicyTrainer,SB3OffPolicyTrainer,SB3HerPolicyTrainer
from reward_surfaces.agents import ExtA2C, ExtPPO, ExtSAC
from reward_surfaces.algorithms import calculate_est_hesh_eigenvalues
from reward_surfaces.agents import RainbowTrainer


def test_curvature(env_fn, trainer):
    # test trainer learning
    saved_files = trainer.train(100,"test_results",save_freq=1000)
    results = trainer.calculate_eigenvalues(100,1.e-5)
    print(results['maxeig'], results['mineig'], results['ratio'])

def discrete_env_fn():
    return gym.make("CartPole-v1")

def continious_env_fn():
    print(gym.make("Pendulum-v1"))
    return gym.make("Pendulum-v1")

if __name__ == "__main__":
    print("testing SB3 SAC curvature")
    test_curvature(continious_env_fn, SB3OffPolicyTrainer(continious_env_fn,ExtSAC("MlpPolicy",continious_env_fn(),device="cuda"), 1, "Pendulum-v1"))
    print("testing SB3 A2C curvature")
    test_curvature(discrete_env_fn, SB3OnPolicyTrainer(discrete_env_fn,ExtA2C("MlpPolicy",discrete_env_fn(),device="cpu",), 1, "CartPole-v1"))
    print("testing SB3 PPO curvature")
    test_curvature(discrete_env_fn, SB3OnPolicyTrainer(discrete_env_fn,ExtPPO("MlpPolicy",discrete_env_fn(),device="cpu") ,1, "CartPole-v1"))
    print("testing Rainbow curvature")
    test_curvature(discrete_env_fn,RainbowTrainer("space_invaders",learning_starts=1000))
