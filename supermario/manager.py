'''

Tem o objetivo de simplificar o treinamento e a execução
'''

import gym
import numpy as np
import os
import shutil
import time
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from callbacks import SaveOnBestTrainingRewardCallback as SaveBestCallBack

import glob


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

        print("Progress:", f'{progress:.1f}%', "Elapsed time:", elapse_time, "End prediction:", end_prediction)


class train_model(object):
    def __init__(self, env, models_dir, log_dir):

        self.env = env
        self.models_dir = models_dir
        self.log_dir = log_dir

        self.rl_map = {}
        self.rl_map["PPO"] = PPO
        self.rl_map["SAC"] = SAC
        self.rl_map["DQN"] = DQN

        self.elapsed_time = 0



    def __set_model_by_name(self, model_name):
        if model_name not in self.rl_map:
            print("Model '", model_name, "' not found")
            return False
        return True


    def prepare_dir(self, dir_path, remove_old_files):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if remove_old_files:
            print("Remove old files from", dir_path)
            for files in os.listdir(dir_path):
                path = os.path.join(dir_path, files)
                try:
                    shutil.rmtree(path)
                except OSError:
                    os.remove(path)



    def train(self, model_name, timesteps = 10000, learning_episodes_number = 30, remove_old_files = True):

        self.__set_model_by_name(model_name)

        log_dir = os.path.join(self.log_dir, model_name)
        model_dir = os.path.join(self.models_dir, model_name)

        self.prepare_dir(log_dir, remove_old_files)
        self.prepare_dir(model_dir, remove_old_files)

        self.env = Monitor(self.env, log_dir) #
        self.env.reset()

        model = self.rl_map[model_name]("MlpPolicy", self.env, verbose = 1, tensorboard_log = log_dir)
        callback = SaveBestCallBack(check_freq=1000, log_dir = log_dir, model_dir = model_dir)


        timer = Timer()
        timer.init_timer()

        for current_episode in range(1, learning_episodes_number):

            model.learn(total_timesteps = timesteps, reset_num_timesteps = False, tb_log_name = model_name, callback = callback)
            save_model_dir = os.path.join(model_dir, str(timesteps * current_episode))
            model.save(save_model_dir)

            timer.mark_end_episode()
            timer.print_cyle_info(learning_episodes_number, current_episode)

        print("Treinamento do", model_name, "finalizado. Tempo Decorrido: ", timer.stringify_time(timer.get_elapsed_time()) )
        self.env.close()
