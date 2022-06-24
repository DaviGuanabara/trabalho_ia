'''

Tem o objetivo de simplificar o treinamento e a execução
'''


import os
import sys
import shutil
import time
import platform

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


def request_sudo():

    if platform.system() is not "Windows":
        euid = os.geteuid()
        if euid != 0:
            print("Script not started as root. Running sudo.")
            args = ['sudo', sys.executable] + sys.argv + [os.environ]
            # the next line replaces the currently-running process with the sudo
            os.execlpe('sudo', *args)
            #os.execlpe('sudo') #, *args)

class Environments(object):

    def __init__(self):
        envs = {}
        envs["SuperMarioBros"] = self.__superMarioBros
        envs["default"] = self.__gen_environment

        self.envs = envs

    def __superMarioBros(self, model_name):
        env = gym_super_mario_bros.make(model_name) #'SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        return env

    def __gen_environment(self, env_name_version):
        return gym.make(env_name_version)

    def get_environment(self, env_name_version):
        env_name = env_name_version.split("-")[0]

        if env_name not in self.envs:
            return self.envs["default"](env_name_version)

        return self.envs[env_name](env_name_version)





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

class Models(object):
    def __init__(self):
        self.rl_map = {}
        self.rl_map["PPO"] = PPO
        self.rl_map["SAC"] = SAC
        self.rl_map["DQN"] = DQN

    def __set_model_by_name(self, model_name):
        if model_name not in self.rl_map:
            print("Model '", model_name, "' not found")
            return False
        return True

    def get_model(self, model_name):
        if model_name not in self.rl_map:
            print("Model '", model_name, "' not found")
            status = False
            return None

        status = True
        return self.rl_map[model_name]



class Trainer(object):


    def __init__(self, env_name_version, model_name, models_dir, log_dir, remove_old_files = True):

        self.__setup_dirs(models_dir, log_dir, model_name, remove_old_files)
        self.__setup_env(env_name_version)
        self.__setup_model(model_name)

        self.model_name = model_name
        self.elapsed_time = 0


    def __setup_dirs(self, models_dir, log_dir, model_name, remove_old_files):
        self.log_dir = os.path.join(log_dir, model_name)
        self.models_dir = os.path.join(models_dir, model_name)

        self.prepare_dir(self.log_dir, remove_old_files)
        self.prepare_dir(self.models_dir, remove_old_files)


    def __setup_env(self, env_name_version):
        self.env = Environments().get_environment(env_name_version)
        self.env = Monitor(self.env, self.log_dir) #
        self.env.reset()


    def __setup_model(self, model_name):
        model = Models().get_model(model_name)
        #Expected one of cpu, cuda, xpu, mkldnn, opengl, opencl, ideep, hip, ve, ort, mlc, xla, lazy, vulkan, meta, hpu device
        device = "cpu"
        #https://stackoverflow.com/questions/1854/python-what-os-am-i-running-on
        #Mac = Darwin
        if platform.system() is "Windows":
            device = "cuda"

        #elif platform.system() is "Darwin":
        #    device = "meta"
        # To Do: tá lançando erro

        self.model = model("MlpPolicy", self.env, verbose = 1, tensorboard_log = self.log_dir, device=device)


    def prepare_dir(self, dir_path, remove_old_files):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if remove_old_files:
            request_sudo()
            #print("Remove old files from", dir_path)
            for files in os.listdir(dir_path):
                path = os.path.join(dir_path, files)
                try:
                    shutil.rmtree(path)
                except OSError:
                    os.remove(path)



    def train(self, timesteps = 10000, learning_episodes_number = 30):


        callback = SaveBestCallBack(check_freq=1000, log_dir = self.log_dir, model_dir = self.models_dir)

        timer = Timer()
        timer.init_timer()

        for current_episode in range(1, learning_episodes_number):

            file_path = os.path.join(self.models_dir, str(timesteps * current_episode))

            self.model.learn(total_timesteps = timesteps, reset_num_timesteps = False, tb_log_name = self.model_name, callback = callback)
            self.model.save(file_path)

            timer.mark_end_episode()
            timer.print_cyle_info(learning_episodes_number, current_episode)

        self.env.close()
        print(f"{self.model_name}'s", "training finished.", "Elapsed time: ", timer.stringify_time(timer.get_elapsed_time()) )

class Executer(object):

    def __init__(self, env_name_version, models_dir, model_name):

        best_model_path = os.path.join(models_dir, model_name, 'best_model')
        print("Collecting best model from:", best_model_path)

        self.env = Environments().get_environment(env_name_version)
        self.model = Models().get_model(model_name).load(best_model_path, self.env)


    def execute(self, episodes):
        for ep in range(episodes):
          obs = self.env.reset()
          done = False
          i = 0
          while not done or i > 100000:
            self.env.render()
            action, _ = self.model.predict(np.copy(obs), deterministic=True)
            obs, reward, done, info = self.env.step(action)

            i += 1

        self.env.close()
