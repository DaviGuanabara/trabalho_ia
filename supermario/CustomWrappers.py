import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from gym import make, ObservationWrapper, wrappers, Wrapper


import csv
import json
import time
from glob import glob
from typing import Dict, List, Optional, Tuple, Union
import pandas

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

class CustomReward(Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        #reward += (info['score'] - self._current_score) / 40.0
        #self._current_score = info['score']
        #if done:
        #    if info['flag_get']:
        #        reward += 350.0
        #    else:
        #        reward -= 50.0

        self.logger.record('reward', reward)
        return state, reward, done, info

class CustomMonitor(Monitor):


    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action
        :param action: the action
        :return: observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        #print("Step in custom_monitor is been executed correctly")

        #print(done)
        #print(self.env._get_done())
        if done:
            print("Episode done")
            print(self.rewards)
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info



class LoadMonitorResultsError(Exception):
    """
    Raised when loading the monitor log fails.
    """

    pass


class ResultsWriter:
    """
    A result writer that saves the data from the `Monitor` class
    :param filename: the location to save a log file, can be None for no log
    :param header: the header dictionary object of the saved csv
    :param reset_keywords: the extra information to log, typically is composed of
        ``reset_keywords`` and ``info_keywords``
    """

    def __init__(
        self,
        filename: str = "",
        header: Optional[Dict[str, Union[float, str]]] = None,
        extra_keys: Tuple[str, ...] = (),
    ):
        if header is None:
            header = {}
        if not filename.endswith(Monitor.EXT):
            if os.path.isdir(filename):
                filename = os.path.join(filename, Monitor.EXT)
            else:
                filename = filename + "." + Monitor.EXT
        # Prevent newline issue on Windows, see GH issue #692
        self.file_handler = open(filename, "wt", newline="\n")
        self.file_handler.write("#%s\n" % json.dumps(header))
        self.logger = csv.DictWriter(self.file_handler, fieldnames=("r", "l", "t") + extra_keys)
        self.logger.writeheader()
        self.file_handler.flush()

    def write_row(self, epinfo: Dict[str, Union[float, int]]) -> None:
        """
        Close the file handler
        :param epinfo: the information on episodic return, length, and time
        """
        if self.logger:
            self.logger.writerow(epinfo)
            self.file_handler.flush()

    def close(self) -> None:
        """
        Close the file handler
        """
        self.file_handler.close()


def get_monitor_files(path: str) -> List[str]:
    """
    get all the monitor files in the given path
    :param path: the logging folder
    :return: the log files
    """
    return glob(os.path.join(path, "*" + Monitor.EXT))


def load_results(path: str) -> pandas.DataFrame:
    """
    Load all Monitor logs from a given directory path matching ``*monitor.csv``
    :param path: the directory path containing the log file(s)
    :return: the logged data
    """
    monitor_files = get_monitor_files(path)
    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(f"No monitor files of the form *{Monitor.EXT} found in {path}")
    data_frames, headers = [], []
    for file_name in monitor_files:
        with open(file_name) as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#"
            header = json.loads(first_line[1:])
            data_frame = pandas.read_csv(file_handler, index_col=None)
            headers.append(header)
            data_frame["t"] += header["t_start"]
        data_frames.append(data_frame)
    data_frame = pandas.concat(data_frames)
    data_frame.sort_values("t", inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame["t"] -= min(header["t_start"] for header in headers)
    return data_frame



class MarioRescale84x84(gym.ObservationWrapper):
    """
    Downsamples/Rescales each frame to size 84x84 with greyscale
    """
    def __init__(self, env=None):
        super(MarioRescale84x84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return MarioRescale84x84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        # image normalization on RBG
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class PixelNormalization(gym.ObservationWrapper):
    """
    Normalize pixel values in frame --> 0 to 1
    """
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class HideHudWrapper(ObservationWrapper):
    """
    Hide the hud changing all pixels in the top of the screen to black.
    Change every observation.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def observation(self, observation):
        observation[0:32, :, :] = 0
        observation[:, :, :] = observation[:, :, :] / 255.
        return observation


# An updated version of the EpisodicLifeEnv wrapper from RLLib which is
# compatible with the SuperMarioBros environments.
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game
        over. Done by DeepMind for the DQN and co. since it helps value
        estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
        self._current_score = 0


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._life
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few fr
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives

        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']
        if done:
            #print('DONE')
            if info['flag_get']:
                reward += 350.0
            else:
                reward -= 0.0
        return obs, reward/100, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped._life
        return obs
