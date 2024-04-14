# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:42:59 2024

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
import seaborn as sns 

os.chdir(r"D:\Georgia Tech\CS7641 Machine Learning\A4")
pd.set_option('display.max_columns', None)

from A4_utils import ModifiedPlanner, ModifiedRL

res = []
n_iters=int(1e6)
gamma = 1.0
test_iters = int(1e4)

np.random.seed(1337)
base_env = gym.make('Blackjack-v1', render_mode=None)
blackjack = BlackjackWrapper(base_env)

planner = ModifiedPlanner(blackjack.P)
start_time = time.time()
vi_V, vi_V_track, vi_pi, vi_iters = planner.value_iteration(gamma=gamma, n_iters=n_iters, theta=1e-10)
vi_time = time.time() - start_time

blackjack.reset(seed=1337)
vi_test_scores = TestEnv.test_env(env=blackjack, n_iters=test_iters, render=False, pi=vi_pi, user_input=False)
vi_test_scores_mean = np.mean(vi_test_scores)
print('VI: Iters=%d | Time=%.5f | TestScore: %.5f' % (vi_iters, vi_time, vi_test_scores_mean))


planner = ModifiedPlanner(blackjack.P)
start_time = time.time()
pi_V, pi_V_track, pi_pi, pi_iters = planner.policy_iteration(gamma=gamma, n_iters=n_iters, theta=1e-10)
pi_time = time.time() - start_time

blackjack.reset(seed=1337)
pi_test_scores = TestEnv.test_env(env=blackjack, n_iters=test_iters, render=False, pi=pi_pi, user_input=False)
pi_test_scores_mean = np.mean(pi_test_scores)

print('PI: Iters=%d | Time=%.5f | TestScore: %.5f' % (pi_iters, pi_time, pi_test_scores_mean))

episodes = 50000
start_time = time.time()
ql_Q, ql_V, ql_pi, ql_Q_track, ql_pi_track = RL(blackjack).q_learning(
               gamma=gamma,
               init_alpha=0.01,
               min_alpha=0.01,
               alpha_decay_ratio=1.0,
               init_epsilon=1.0,
               min_epsilon=0.2,
               epsilon_decay_ratio=0.99,
               #n_episodes=1000000)
               n_episodes=episodes)
ql_time = time.time() - start_time
blackjack.reset(seed=1337)
ql_test_scores = TestEnv.test_env(env=blackjack, n_iters=test_iters, render=False, pi=ql_pi, user_input=False)
ql_test_scores_mean = np.mean(ql_test_scores)
print('QL: Iters=%d | Time=%.2f | TestScore: %.5f' % (episodes, ql_time, ql_test_scores_mean))

print(vi_pi==pi_pi, vi_pi==ql_pi)


# Charting
plt.figure()
plt.plot(list(range(1,vi_iters+1)), [np.mean(v) for v in vi_V_track[:vi_iters]], '.-', color="b")
plt.title("Blackjack V(Mean) for Value Iteration")
plt.xlabel("Iteration")
plt.ylabel("V(Mean)")           
plt.savefig("charts/Blackjack VI.png")

plt.figure()
plt.plot(list(range(1,pi_iters+1)), [np.mean(v) for v in pi_V_track[:pi_iters]], '.-', color="b")
plt.title("Blackjack V(Mean) for Policy Iteration")
plt.xlabel("Iteration")
plt.ylabel("V(Mean)")           
plt.savefig("charts/Blackjack PI.png")

plt.figure()
plt.plot(list(range(1,len(ql_Q_track)+1)), [np.mean(q) for q in ql_Q_track], '.-', color="b")
plt.title("Blackjack Q(Mean) for Q Learning")
plt.xlabel("Iteration")
plt.ylabel("Q(Mean)")           
plt.savefig("charts/Blackjack QL.png")
    

plt.figure(figsize=(10,12))
plt.title("Blackjack\n State Values for Value Iteration")
hm = sns.heatmap(data=vi_V.reshape(29,10),annot=True) 
plt.savefig("charts/Blackjack HM VI.png")

plt.figure(figsize=(10,12))
plt.title("Blackjack\n State Values for Policy Iteration")
hm = sns.heatmap(data=pi_V.reshape(29,10),annot=True) 
plt.savefig("charts/Blackjack HM PI.png")

plt.figure(figsize=(10,12))
plt.title("Blackjack\n State Values for Q learning")
hm = sns.heatmap(data=ql_V.reshape(29,10),annot=True) 
plt.savefig("charts/Blackjack HM QL.png")




