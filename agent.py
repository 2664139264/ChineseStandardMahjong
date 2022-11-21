from abc import ABCMeta, abstractmethod  

from multiagent_env import MultiAgentEnv

class Agent(metaclass=ABCMeta):

    @abstractmethod
    def select_action(self, obs:MultiAgentEnv.ObservationType) -> MultiAgentEnv.ActionType:
        raise NotImplementedError