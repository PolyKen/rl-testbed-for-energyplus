"""
Helpers for script run_energyplus.py.
"""

import os
import gym
from baselines import logger
from baselines_energyplus.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from mpi4py import MPI
import glob


def make_energyplus_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for EnergyEnv
    """
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir())
    env.seed(seed)
    return env


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def energyplus_arg_parser():
    """
    Create an argparse.ArgumentParser for run_energypl.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', '-e', help='environment ID', type=str, default='EnergyPlus-v0')
    parser.add_argument('--seed', '-s', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', '-n', type=int, default=int(100000))
    parser.add_argument('--save-interval', type=int, default=int(0))
    parser.add_argument('--model-pickle', help='model pickle', type=str, default='')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='')
    parser.add_argument('--learn', '-l', help='name of some function in algorithms/learning_methods.py', type=str,
                        default='default_learn')
    parser.add_argument('--policy', '-p', help='name of some class in algorithms/policies.py', type=str,
                        default='DefaultMlpPolicy')
    return parser


def energyplus_locate_log_dir(index=0):
    pat = energyplus_logbase_dir() + '/openai-????-??-??-??-??-??-??????*/progress.csv'
    files = [(f, os.path.getmtime(f)) for f in glob.glob(pat)]
    newest = sorted(files, key=lambda files: files[1])[-(1 + index)][0]
    dir = os.path.dirname(newest)
    print('energyplus_locate_log_dir: {}'.format(dir))
    return dir


def energyplus_logbase_dir():
    logbase_dir = os.getenv('ENERGYPLUS_LOGBASE')
    if logbase_dir is None:
        logbase_dir = '/tmp'
    return logbase_dir
