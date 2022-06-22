'''

Utils
'''

import gym
import numpy as np
import os
import time
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from callbacks import SaveOnBestTrainingRewardCallback as SaveBestCallBack
import time

class training_model(object):
    def __init__(self, env, models_dir, logdir):

        self.env = env
        self.models_dir = models_dir
        self.logdir = logdir

        self.rl_map = {}
        self.rl_map["PPO"] = PPO
        self.rl_map["SAC"] = SAC
        self.rl_map["DQN"] = DQN




    def train(self, model_name, timesteps=10000):

        #os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

        if model_name not in self.rl_map:
            print("Model '", model_name, "' not found")
            return;

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)


        #env = gym.make(self.gym_name)
        #self.env = Monitor(self.env, self.logdir)
        self.env.reset()

        model = self.rl_map[model_name]("MlpPolicy", self.env, verbose=1, tensorboard_log=self.logdir)
        callback = SaveBestCallBack(check_freq=1000, log_dir = f"{self.logdir}/{model_name}", model_dir = f"{self.models_dir}/{model_name}")

        print("Starting training of", model_name)
        initial_time = time.time()
        for i in range(1, 30):

            start = time.time()
            timestamp = timesteps * i
            print("Learning. Step between", timesteps * (i-1), "and", timestamp)
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=model_name, callback=callback)

            print("Saving in", f"{self.models_dir}/{model_name}/{timestamp}")
            model.save(f"{self.models_dir}/{model_name}/{timestamp}")

            end = time.time()

            td = (end - initial_time)
            pt = ((30 - i) / i) * td
            print("Concluído:", f'{i * 100/30:.1f}%', "Tempo decorrido:", f'{td/60: .1f}' , "minutos. Previsão de término:", f'{pt/60: .1f}' , "minutos")

        print("Treinamento do", model_name, "finalizado. Tempo Decorrido: ", f'{(end - initial_time)/60: .1f} minutos' )
        self.env.close()
