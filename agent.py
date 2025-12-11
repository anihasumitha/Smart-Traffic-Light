# agent.py
import random
import numpy as np

class QAgent:
    """
    Small Q-learning agent with discrete states:
      cars_north: 0..5
      cars_south: 0..5
      light: 0/1
    Actions: 0 (keep), 1 (switch)
    """

    def __init__(self, lr=0.1, gamma=0.9, epsilon=0.1):
        # Q-table shape: (6,6,2,2)
        self.q = np.zeros((6, 6, 2, 2), dtype=float)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        cn, cs, light = state
        # epsilon-greedy: random action with probability epsilon
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        # otherwise choose best action from Q-table
        return int(np.argmax(self.q[cn, cs, light]))

    def update(self, state, action, reward, next_state):
        cn, cs, light = state
        ncn, ncs, nlight = next_state
        # Q-learning update rule
        best_next = np.max(self.q[ncn, ncs, nlight])
        old_val = self.q[cn, cs, light, action]
        self.q[cn, cs, light, action] = old_val + self.lr * (reward + self.gamma * best_next - old_val)
