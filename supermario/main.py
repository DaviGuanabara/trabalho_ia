import os
import time

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import utils


env_name = 'SuperMarioBros-v0' #"BipedalWalker-v3"
log_dir = "logs" #+ env_name
model_dir = "models"# + env_name
timesteps = 1000
models = ["PPO"]


#env = gym.make("BipedalWalker-v3")
#env = Monitor(env, log_dir)
#env.reset()


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = Monitor(env, log_dir)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#env = Monitor(env, self.logdir)
env.reset()

training_model = utils.training_model(env, model_dir, log_dir)

for model_name in models:
    training_model.train(model_name, timesteps=timesteps)

#training_model = utils.training_model("BipedalWalker-v3", "models/PPO", f"logs/PPO-{int(time.time())}")
#training_model.train(PPO)
#plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "SAC BipedalWalker")
#plt.show()
