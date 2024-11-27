import math
import numpy as np
import random


class MAB:
    def __init__(self, n_arms, c=1):
        self.n_arms = n_arms
        self.mus = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.tot_rounds = 0
        # hyper parameters
        self.c = c
        self.epsilon = 0.15
        # auxiliary values
        self.min_reward, self.max_reward = -math.log(0.95), -math.log(0.05)
        self.max_min_reward_dif = self.max_reward - self.min_reward
        self.initialized = False   #

    def update(self, arm, reward):
        """ update mean reward of given arm according to observed reward """
        self.tot_rounds += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.mus[arm] = ((n - 1) / n) * self.mus[arm] + (1 / n) * reward

    " Choose Arm Methods "

    def ucb_choose_arm(self):
        """ choose an arm based on the UCB formula """
        ucb = lambda mu_i, n_i: mu_i + self.c * math.sqrt((math.log(self.tot_rounds) / n_i))
        ucb_scores = [ucb(self.mus[i], self.counts[i]) for i in range(self.n_arms)]
        return np.argmax(ucb_scores)

    def epsilon_greedy_choose_arm(self):
        """ choose an arm based on epsilon-greedy approch """
        if random.random() <= self.epsilon:
            return random.randint(0, self.n_arms-1)
        else:
            return np.argmax(self.mus)

    " Reward Methods "

    def ce_reward(self, pred_probs, real):
        """ ross entropy reward """
        prob_given_to_real = pred_probs[real]
        # cap to [0.05, 0.95]
        prob = max(min(0.95, prob_given_to_real), 0.05)
        # compute and normalize reward
        reward = (-math.log(prob) - self.min_reward) / self.max_min_reward_dif
        return reward

    def vx_reward(self, pred_probs, real):
        """ binary reward"""
        prob_given_to_real = pred_probs[real]
        if prob_given_to_real > 0.5:
            return 0.001
        else:
            return 1

    def con_reward(self, pred_probs, real):
        prob_given_to_real = pred_probs[real]
        return 1-prob_given_to_real


    " Auxiliary Methods "

    def get_ranks(self):
        """ get UCB scores for each arm """
        ucb = lambda mu_i, n_i, tot: mu_i + self.c * math.sqrt((math.log(self.tot_rounds) / n_i))
        ucb_scores = [ucb(self.mus[i], self.counts[i], self.tot_rounds) for i in range(self.n_arms)]
        return ucb_scores