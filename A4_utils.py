# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:44:15 2024

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

class ModifiedPlanner(Planner):
    def __init__(self, P):
        super().__init__(P)
        
    def value_iteration(self, gamma=1.0, n_iters=1000, theta=1e-10):
        # Source: https://github.com/jlm429/bettermdptools/blob/master/bettermdptools/algorithms/planner.py
        # Overriding function to make some minor modifications
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        i = 0
        converged = False
        while i < n_iters-1 and not converged:
            i += 1
            Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
            for s in range(len(self.P)):
                for a in range(len(self.P[s])):
                    for prob, next_state, reward, done in self.P[s][a]:
                        Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
                converged = True
            V = np.max(Q, axis=1)
            V_track[i] = V
        if not converged:
            print("Max iterations reached before convergence.  Check theta and n_iters.  ")

        pi = {s:a for s, a in enumerate(np.argmax(Q, axis=1))}
        iters = i + 1
        return V, V_track, pi, iters

    def policy_iteration(self, gamma=1.0, n_iters=50, theta=1e-10):
        # Source: https://github.com/jlm429/bettermdptools/blob/master/bettermdptools/algorithms/planner.py
        # Overriding function to make some minor modifications
        random_actions = np.random.choice(tuple(self.P[0].keys()), len(self.P))

        pi = {s: a for s, a in enumerate(random_actions)}
        # initial V to give to `policy_evaluation` for the first time
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        i = 0
        converged = False
        while i < n_iters-1 and not converged:
            i += 1
            old_pi = pi
            V = self.policy_evaluation(pi, V, gamma, theta)
            V_track[i] = V
            pi = self.policy_improvement(V, gamma)
            if old_pi == pi:
                converged = True
        if not converged:
            print("Max iterations reached before convergence.  Check n_iters.")
        iters = i + 1
        return V, V_track, pi, iters


class ModifiedRL(RL):
    def __init__(self, env):
        super().__init__(env)

    def q_learning(self,
                   nS=None,
                   nA=None,
                   convert_state_obs=lambda state: state,
                   gamma=.99,
                   init_alpha=0.5,
                   min_alpha=0.01,
                   alpha_decay_ratio=0.5,
                   init_epsilon=1.0,
                   min_epsilon=0.1,
                   epsilon_decay_ratio=0.9,
                   n_episodes=10000):

        if nS is None:
            nS=self.env.observation_space.n
        if nA is None:
            nA=self.env.action_space.n
        pi_track = []
        Q = np.zeros((nS, nA), dtype=np.float64)
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        # Explanation of lambda:
        # def select_action(state, Q, epsilon):
        #   if np.random.random() > epsilon:
        #       return np.argmax(Q[state])
        #   else:
        #       return np.random.randint(len(Q[state]))
        select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
            if np.random.random() > epsilon \
            else np.random.randint(len(Q[state]))
        alphas = RL.decay_schedule(init_alpha,
                                min_alpha,
                                alpha_decay_ratio,
                                n_episodes)
        epsilons = RL.decay_schedule(init_epsilon,
                                  min_epsilon,
                                  epsilon_decay_ratio,
                                  n_episodes)
        for e in tqdm(range(n_episodes), leave=False):
            self.callbacks.on_episode_begin(self)
            self.callbacks.on_episode(self, episode=e)
            state, info = self.env.reset()
            done = False
            state = convert_state_obs(state)
            while not done:
                if self.render:
                    warnings.warn("Occasional render has been deprecated by openAI.  Use test_env.py to render.")
                action = select_action(state, Q, epsilons[e])
                #next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                @functools.lru_cache(maxsize=None)
                def getProbs(state, action):
                    return np.array(self.env.P[state][action])[:,0]
                   
                idx = np.random.choice(range(len(self.env.P[state][action])), p=getProbs(state, action))
                _, next_state, reward, terminated = self.env.P[state][action][idx]
                #if truncated:
                #    warnings.warn("Episode was truncated.  Bootstrapping 0 reward.")
                #done = terminated or truncated
                done = terminated
                self.callbacks.on_env_step(self)
                next_state = convert_state_obs(next_state)
                td_target = reward + gamma * Q[next_state].max() * (not done)
                td_error = td_target - Q[state][action]
                Q[state][action] = Q[state][action] + alphas[e] * td_error
                state = next_state
            Q_track[e] = Q
            pi_track.append(np.argmax(Q, axis=1))
            self.render = False
            self.callbacks.on_episode_end(self)

        V = np.max(Q, axis=1)

        pi = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}
        return Q, V, pi, Q_track, pi_track