# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:06:48 2024

@author: User
"""

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots
from bettermdptools.algorithms.rl import RL
import time
import six
import copy
from tqdm import tqdm
import warnings
import functools
import os
import matplotlib.pyplot as plt

from A4_utils import ModifiedPlanner, ModifiedRL

os.chdir(r"D:\Georgia Tech\CS7641 Machine Learning\A4")
pd.set_option('display.max_columns', None)


gym.envs.register(
    id='FrozenLake-v1',
    entry_point='gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv',
    max_episode_steps=10000 # overriding default of 100
)


def modifyTransitionMatrix(P):
    modified_P = copy.deepcopy(P)
    
    reward_mapping = {'S':-0.01,
                      'F':-0.01,
                      'H':-1.0,
                      'G':100.0,}
    joined_amap = ''.join(amap)
    for state, action_dict in six.iteritems(P):
        for action, tup_list in six.iteritems(action_dict):
            new_tup_list = copy.deepcopy(tup_list)
            for j, tup in enumerate(tup_list):
                new_tup_list[j] = (tup[0], tup[1], reward_mapping.get(joined_amap[tup[1]]), tup[3])
            modified_P[state][action] = new_tup_list
    return modified_P

def printMap( amap, pi):
    actions_map = {0:'<',
                   1:'v',
                   2:'>',
                   3:'^',}


    pi = np.asarray(list(pi.values())).reshape((len(amap), len(amap)))
    to_print = [[*x] for x in amap]
    for i, r in enumerate(amap):
        for j, s in enumerate(r):
            if s in ['S','F']:
                to_print[i][j] = actions_map.get(pi[i][j])

    print(pd.DataFrame(to_print))

frozen_lake_sizes = [ 4, 8, 12, 16]

ql_episode_map = {4:5000,
                  8:10000,
                  12:30000,
                  16:100000}

res = []
n_iters=int(1e5)
gamma = 1.0
for n in frozen_lake_sizes:
    size=(n,n)
    np.random.seed(1337)
    amap = generate_random_map(size=n)
    frozen_lake = gym.make('FrozenLake-v1', desc=amap)
    modified_P = modifyTransitionMatrix(frozen_lake.P)
    frozen_lake.P = modified_P
    #modified_P = frozen_lake.P
    state_size = len(modified_P)
    
    planner = ModifiedPlanner(modified_P)
    start_time = time.time()
    vi_V, vi_V_track, vi_pi, vi_iters = planner.value_iteration(gamma=gamma, n_iters=n_iters, theta=1e-5)
    vi_time = time.time() - start_time

    printMap(amap, vi_pi)
    frozen_lake.reset(seed=1337)
    vi_test_scores = TestEnv.test_env(env=frozen_lake, desc=amap, n_iters=100, render=False, pi=vi_pi, user_input=False)
    vi_test_scores_mean = np.mean(vi_test_scores)
    print('VI(%dx%d): StateSize=%s | Iters=%d | Time=%.2f | TestScore: %.5f' % (n, n, state_size, vi_iters, vi_time, vi_test_scores_mean))
    
    planner = ModifiedPlanner(modified_P)
    start_time = time.time()
    pi_V, pi_V_track, pi_pi, pi_iters = planner.policy_iteration(gamma=gamma, n_iters=n_iters, theta=1e-5)
    pi_time = time.time() - start_time
    #Plots.values_heat_map(V, "Frozen Lake\nValue Policy State Values", size)
    printMap(amap, pi_pi)
    frozen_lake.reset(seed=1337)
    pi_test_scores = TestEnv.test_env(env=frozen_lake, desc=amap, n_iters=100, render=False, pi=pi_pi, user_input=False)
    pi_test_scores_mean = np.mean(pi_test_scores)
    
    print('PI(%dx%d): StateSize=%s | Iters=%d | Time=%.2f | TestScore: %.5f' % (n, n, state_size, pi_iters, pi_time, pi_test_scores_mean))
    
    start_time = time.time()
    ql_Q, ql_V, ql_pi, ql_Q_track, ql_pi_track = ModifiedRL(frozen_lake).q_learning(
                   gamma=gamma,
                   init_alpha=0.1,
                   min_alpha=0.1,
                   alpha_decay_ratio=0.99,
                   init_epsilon=1.0,
                   min_epsilon=0.1,
                   epsilon_decay_ratio=0.99,
                   n_episodes=ql_episode_map.get(n))
    ql_time = time.time() - start_time
    printMap(amap, ql_pi)
    frozen_lake.reset(seed=1337)
    ql_test_scores = TestEnv.test_env(env=frozen_lake, desc=amap, n_iters=100, render=False, pi=ql_pi, user_input=False)
    ql_test_scores_mean = np.mean(ql_test_scores)
    print('QL(%dx%d): StateSize=%s | Iters=%d | Time=%.2f | TestScore: %.5f' % (n, n, state_size, ql_episode_map.get(n), ql_time, ql_test_scores_mean))


    # Charting
    plt.figure()
    plt.plot(list(range(1,vi_iters+1)), [np.mean(v) for v in vi_V_track[:vi_iters]], '.-', color="b")
    plt.title("FrozenLake(%dx%d) V(Mean) for Value Iteration" % (n,n))
    plt.xlabel("Iteration")
    plt.ylabel("V(Mean)")           
    plt.savefig("charts/FrozenLake(%dx%d) VI.png" % (n, n))

    plt.figure()
    plt.plot(list(range(1,pi_iters+1)), [np.mean(v) for v in pi_V_track[:pi_iters]], '.-', color="b")
    plt.title("FrozenLake(%dx%d) V(Mean) for Policy Iteration" % (n,n))
    plt.xlabel("Iteration")
    plt.ylabel("V(Mean)")           
    plt.savefig("charts/FrozenLake(%dx%d) PI.png" % (n, n))

    plt.figure()
    plt.plot(list(range(1,len(ql_Q_track)+1)), [np.mean(q) for q in ql_Q_track], '.-', color="b")
    plt.title("FrozenLake(%dx%d) Q(Mean) for Q Learning" % (n,n))
    plt.xlabel("Iteration")
    plt.ylabel("Q(Mean)")           
    plt.savefig("charts/FrozenLake(%dx%d) QL.png" % (n, n))
