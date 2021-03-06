'''
Resolução de problemas

1. Problema
2022-06-23 22:10:12.715 Python[7483:238807] ApplePersistenceIgnoreState: Existing state will not be touched. New state will be written to /var/folders/v0/2k2s91hs7g3czdr66q9zyr40
0000gn/T/org.python.python.savedState

1. Solução
Davis-MacBook-Pro:supermario daviaragao$ defaults write org.python.python ApplePersistenceIgnoreState NO


Executar o TensorBoard:

supermario$ tensorboard --logdir=logs
'''

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

import manager

#================================================================
# perfils
#================================================================

perfils = {}
perfils["SuperMario"] = {}
perfils["SuperMario"]["env_name_version"] = 'SuperMarioBros-v3'
perfils["SuperMario"]["models"] = ["DQN", "PPO"]
perfils["SuperMario"]["learning_rate"] = 0.01

#CarRacing is a discrete action space environment
perfils["CarRacing"] = {}
perfils["CarRacing"]["env_name_version"] = 'CarRacing-v0' #CarRacing-v1 is deprecated
perfils["CarRacing"]["models"] = ["SAC", "PPO"]
perfils["CarRacing"]["learning_rate"] = 0.01

perfils["BipedalWalker"] = {}
perfils["BipedalWalker"]["env_name_version"] = 'BipedalWalker-v3'
perfils["BipedalWalker"]["models"] = ["SAC", "PPO"]
#perfils["BipedalWalker"]["models"] = ["PPO"]
perfils["BipedalWalker"]["learning_rate"] = 10 ** -6

#================================================================
# Configurações
#================================================================

perfil = perfils["BipedalWalker"]

env_name_version = perfil["env_name_version"]
models = perfil["models"]


enable_learning = True
enable_executing = False

'''
Configurações Padrão
'''

log_dir = "logs"
models_dir = "models"
model_filename = 'best_model'
#model_filename = 'best_model'
#não colocar menos do que 1000, por algum motivo não salva o melhor modelo.
#timesteps = 100000

timesteps = 500_000
learning_episodes_number = 5
learning_rate = perfil["learning_rate"]
#================================================================
# Execução do código.
#================================================================

if enable_learning:

    for model_name in models:
        trainer = manager.Trainer(env_name_version, model_name, models_dir, log_dir, learning_rate=learning_rate)
        trainer.train(timesteps=timesteps, learning_episodes_number=learning_episodes_number)


if enable_executing:
    for model_name in models:
        print("Executing", env_name_version, "on", model_name)
        executer = manager.Executer(env_name_version, models_dir, model_name, model_filename=model_filename)
        executer.execute(10)
