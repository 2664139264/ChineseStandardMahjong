from typing import Any, Iterable
from abc import ABCMeta, abstractmethod

class SamplePool(metaclass=ABCMeta):
    SampleType = Any
    
    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def save_sample(self, sample:SampleType, tag:str):
        raise NotImplementedError
    
    @abstractmethod
    def load_batch(self, batch_size:int) -> Iterable[SampleType]:
        raise NotImplementedError    