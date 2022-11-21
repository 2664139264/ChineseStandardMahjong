
import re
import numpy as np
from agent import Agent

class RandomMahjongAgent(Agent):
    def __init__(self):
        super().__init__()
    
    def select_action(self, obs):
        action_space = obs['action_space']
        np.random.shuffle(action_space)
        selected_action = action_space[0]
        for action in action_space:
            if re.search(r'Hu|Gang|Peng|Chi', action) is not None:
                selected_action = action
                break
        return selected_action