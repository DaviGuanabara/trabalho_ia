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
from CustomWrappers import CustomReward, CustomMonitor



class Environments(object):

    def __init__(self):
        envs = {}
        envs["SuperMarioBros"] = (self.__superMarioBros, {"policy": "CnnPolicy"})
        envs["default"] = (self.__gen_default_environment, {"policy": "MlpPolicy"})

        self.envs = envs

    def __superMarioBros(self, model_name, log_dir, enable_monitor=False):
        env = gym_super_mario_bros.make(model_name ) #'SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        if enable_monitor:
            env = Monitor(env, log_dir)
            #env = CustomMonitor(env, log_dir)

        env = GrayScaleObservation(env, keep_dim=True)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 6, channels_order='last')

        return env

    def __gen_default_environment(self, env_name_version, log_dir, enable_monitor=False):
        print("Entrou no gen default")
        env = gym.make(env_name_version)

        if enable_monitor:
            print("Ativou o monitor")
            env = Monitor(env, log_dir)

        print("Retornar o ambiente")
        return env

    def get_environment(self, env_name_version, log_dir, enable_monitor=False):
        print("env_name", env_name_version)
        env_name = env_name_version.split("-")[0]


        if env_name not in self.envs:
            print("env_name not in envs")
            env_name = "default"

        print("env_name", env_name)
        return self.envs[env_name][0](env_name_version, log_dir, enable_monitor=enable_monitor)

    def get_env_info(self, env_name_version):
        env_name = env_name_version.split("-")[0]

        if env_name not in self.envs:
            env_name = "default"

        return self.envs[env_name][1]


class Models(object):
    def __init__(self):
        self.rl_map = {}
        self.rl_map["PPO"] = (self.__create_ppo, self.__load_ppo)
        self.rl_map["SAC"] = (self.__create_sac, self.__load_sac)
        self.rl_map["DQN"] = (self.__create_dqn, self.__load_dqn)

    def __create_ppo(self, env, env_info={"policy": "MlpPolicy"}, tensorboard_log=None, learning_rate="0.0003", n_steps=2048, create_eval_env=False):
        rl_algorithm = PPO
        return rl_algorithm(env_info["policy"], env, verbose=1, tensorboard_log=tensorboard_log, learning_rate=learning_rate, n_steps=n_steps, create_eval_env=create_eval_env)

    def __create_dqn(self, env, env_info={"policy": "MlpPolicy"}, tensorboard_log=None, learning_rate="0.0001", n_steps=1024, create_eval_env=False):
        rl_algorithm = DQN
        return rl_algorithm(env_info["policy"], env, verbose=1, tensorboard_log=tensorboard_log, learning_rate=learning_rate, create_eval_env=create_eval_env)

    def __create_sac(self, env, env_info={"policy": "MlpPolicy"}, tensorboard_log=None, learning_rate="0.0003", n_steps=1024, create_eval_env=False):
        rl_algorithm = SAC
        return rl_algorithm(env_info["policy"], env, verbose=1, tensorboard_log=tensorboard_log, learning_rate=learning_rate, create_eval_env=create_eval_env)



    def __load_ppo(self, model_storage_path, env):
        rl_algorithm = PPO
        return rl_algorithm.load(model_storage_path, env)

    def __load_dqn(self, model_storage_path, env):
        rl_algorithm = DQN
        return rl_algorithm.load(model_storage_path, env)

    def __load_sac(self, model_storage_path, env):
        rl_algorithm = SAC
        return rl_algorithm.load(model_storage_path, env)

    def create_model(self, model_name, env, env_info={"policy": "MlpPolicy"}, tensorboard_log="logs", learning_rate="0.000001", n_steps=1024, create_eval_env=True):

        if model_name not in self.rl_map:
            print("Model '", model_name, "' not found")
            return None

        return self.rl_map[model_name][0](env, env_info, tensorboard_log=tensorboard_log, learning_rate=learning_rate, n_steps=n_steps, create_eval_env=create_eval_env)

    def load_model(self, model_name, model_storage_path, env):

        if model_name not in self.rl_map:
            print("Model '", model_name, "' not found")
            return None

        return self.rl_map[model_name][1](model_storage_path, env)




class Timer(object):

    def init_timer(self):
        self.init = time.time()

    def stringify_time(self, time):

        if int(time / 3600) > 0:
            return f'{time / 3600:.1f}' + ' hour(s)'

        if int(time / 60) > 0:
            return f'{time / 60:.1f}' + ' minute(s)'

        return f'{time:.1f}' + ' second(s)'

    def mark_begin_episode(self):
        self.begin = time.time()

    def mark_end_episode(self):
        self.end = time.time()

    def get_elapsed_time(self):
        return self.end - self.init

    def get_end_prediction(self, number_episodes, current_episode):
        return ((number_episodes - current_episode) / current_episode) * self.get_elapsed_time()

    def print_cyle_info(self, number_episodes, current_episode, dimension = 'minutes'):

        divisor = 1

        if dimension == 'minutes':
            divisor = 60

        if dimension == 'hour':
            divisor = 3600

        progress = current_episode * 100/number_episodes

        elapse_time = self.stringify_time(self.get_elapsed_time())
        end_prediction = self.stringify_time(self.get_end_prediction(number_episodes, current_episode))

        print("Progress:", f'{progress:.1f}%', "- Elapsed time:", elapse_time, "- Ending in:", end_prediction)
