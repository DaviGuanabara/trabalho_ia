import os
import sys
import shutil
import time

import gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from callbacks import SaveOnBestTrainingRewardCallback as SaveBestCallBack

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


from factory import Environments, Timer, Models
from utils import request_sudo



env_name_version = 'SuperMarioBros-v3'
#env_name_version = "BipedalWalker-v3"

env_manager = Environments()
env = env_manager.get_environment(env_name_version, "logs")
env.reset()
env_info = env_manager.get_env_info(env_name_version)
print(env_info["policy"])
#self.model = Models().get_model(model_name).load(best_model_path, self.env)
