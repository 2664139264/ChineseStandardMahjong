from typing import Any
from abc import ABCMeta, abstractmethod

class Trainer(metaclass=ABCMeta):
    
    EvaluateResultType = Any
    LogContentType = Any

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> EvaluateResultType:
        raise NotImplementedError
    
    @abstractmethod
    def write_log(self, content:LogContentType):
        raise NotImplementedError