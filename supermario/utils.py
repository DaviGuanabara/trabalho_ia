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



def request_sudo():
    euid = os.getlogin()
    if euid != 0:
        print("Script not started as root. Running sudo.")
        args = ['sudo', sys.executable] + sys.argv + [os.environ]
        # the next line replaces the currently-running process with the sudo
        os.execlpe('sudo', *args)
        #os.execlpe('sudo') #, *args)

def prepare_dir(dir_path, remove_old_files):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if remove_old_files:
        #request_sudo()
        #print("Remove old files from", dir_path)
        for files in os.listdir(dir_path):
            path = os.path.join(dir_path, files)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)
