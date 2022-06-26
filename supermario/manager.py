'''

Tem o objetivo de simplificar o treinamento e a execução
'''


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
from utils import request_sudo, prepare_dir
from CustomWrappers import CustomReward



class Trainer(object):


    def __init__(self, env_name_version, model_name, models_dir, log_dir, remove_old_files = True):

        self.log_dir, self.models_dir = self.__setup_dirs(models_dir, log_dir, model_name, remove_old_files)

        self.env, env_info = self.__setup_env(env_name_version)
        self.model = self.__setup_model(model_name, self.env, env_info, self.log_dir)

        self.model_name = model_name
        self.elapsed_time = 0


    def __setup_dirs(self, models_dir, log_dir, model_name, remove_old_files):
        log_dir = os.path.join(log_dir, model_name)
        models_dir = os.path.join(models_dir, model_name)

        prepare_dir(log_dir, remove_old_files)
        prepare_dir(models_dir, remove_old_files)

        return log_dir, models_dir

    def __setup_env(self, env_name_version):

        env_manager = Environments()
        env = env_manager.get_environment(env_name_version, self.log_dir, enable_monitor=True)
        env_info = env_manager.get_env_info(env_name_version)

        env.reset()

        return env, env_info


    def __setup_model(self, model_name, env, env_info, log_dir, learning_rate=0.000001, n_steps=1024):

        return Models().create_model(model_name, env, env_info, tensorboard_log=log_dir, learning_rate=learning_rate, n_steps=n_steps, create_eval_env=True)



    def train(self, timesteps = 10000, learning_episodes_number = 40):


        saveBestCallBack = SaveBestCallBack(check_freq=1000, log_dir = self.log_dir, model_dir = self.models_dir)

        timer = Timer()
        timer.init_timer()

        for current_episode in range(1, learning_episodes_number):

            file_path = os.path.join(self.models_dir, str(timesteps * current_episode))

            # Pass reset_num_timesteps=False to continue the training curve in tensorboard
            # By default, it will create a new curve
            self.model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=f"{self.model_name}", callback=saveBestCallBack)
            self.model.save(file_path)

            timer.mark_end_episode()
            timer.print_cyle_info(learning_episodes_number, current_episode)

        #self.env.ResultsWriter()
        self.env.close()
        print(f"{self.model_name}'s", "training finished.", "Elapsed time: ", timer.stringify_time(timer.get_elapsed_time()) )

class Executer(object):

    def __init__(self, env_name_version, models_dir, model_name, model_filename='best_model', log_dir=''):




        env_manager = Environments()
        env = env_manager.get_environment(env_name_version, log_dir)
        env_info = env_manager.get_env_info(env_name_version)
        #self.model = Models().get_model(model_name).load(best_model_path, self.env)

        model_path = os.path.join(models_dir, model_name, model_filename)
        model = Models().load_model(model_name, model_path, env)

        self.env = env
        self.model = model

    def execute(self, episodes):
        for ep in range(episodes):
          obs = self.env.reset()
          done = False
          i = 0
          while not done or i > 100000:

            action, _ = self.model.predict(np.copy(obs))
            obs, reward, done, info = self.env.step(action)

            self.env.render()
            i += 1

        self.env.close()
