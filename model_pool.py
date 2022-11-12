from typing import Any, Iterable
from abc import ABCMeta, abstractmethod

class ModelPool(metaclass=ABCMeta):
    ModelType = Any
    ModelHandlerType = Any
    DistributionType = Any
    
    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def save_model(self, model:ModelType, tag:str) -> ModelHandlerType:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, handler:ModelHandlerType) -> ModelType:
        raise NotImplementedError

    @abstractmethod
    def sample_model(self, n:int, dist:DistributionType) -> Iterable[ModelType]:
        raise NotImplementedError