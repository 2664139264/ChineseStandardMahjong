from typing import Any
from abc import ABCMeta, abstractmethod  

class Agent(metaclass=ABCMeta):

    ObservationType = Any
    ActionType = Any
    LearnableDataType = Any
    LearnReturnType = Any
    SelectActionModeType = Any
    LogContentType = Any
    
    @abstractmethod
    def load_model(self, path:str):
        raise NotImplementedError
        
    @abstractmethod
    def save_model(self, path:str):
        raise NotImplementedError
    
    @abstractmethod
    def write_log(self, content:LogContentType):
        raise NotImplementedError

    @abstractmethod
    def select_action(self, obs:ObservationType, mode:SelectActionModeType) -> ActionType:
        raise NotImplementedError
    
    @abstractmethod
    def learn(self, data:LearnableDataType) -> LearnReturnType:
        raise NotImplementedError