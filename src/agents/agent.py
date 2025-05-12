import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
        # Initialize Q-table with zeros
        self.q_table = {}
        
    
    def get_action(self, state):
        state_key = self._get_state_key(state)
        
        # Exploration: choose random action
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        
        # Exploitation: choose best action from Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update rule
        old_value = self.q_table[state_key][action]
        if done:
            next_max = 0
        else:
            next_max = np.max(self.q_table[next_state_key])
        
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        
        self.q_table[state_key][action] = new_value
        
        # Decay exploration rate
        if done:
            self.exploration_rate *= self.exploration_decay
    
    def save_q_table(self, filename):
        np.save(filename, self.q_table)
    
    def load_q_table(self, filename):
        self.q_table = np.load(filename, allow_pickle=True).item()



    def _get_state_key(self, state):
        return tuple(state)
    