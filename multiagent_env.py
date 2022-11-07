from ast import Not
from typing import Any, Tuple, Dict, List
from abc import ABCMeta, abstractmethod  

class MultiAgentEnv(metaclass=ABCMeta):

    StateType = Any
    ObservationType = Any
    ActionType = Any
    StepInfoType = Any
    CloseInfoType = Any
    PlayerIDType = int

    @property
    @abstractmethod
    def n_players(self) -> int:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def active_player(self) -> PlayerIDType:
        raise NotImplementedError

    @abstractmethod
    def state(self) -> StateType:
        raise NotImplementedError

    @abstractmethod
    def observation(self) -> ObservationType:
        raise NotImplementedError

    @abstractmethod
    def history(self) -> List[Tuple[StateType, PlayerIDType, ActionType]]:
        raise NotImplementedError

    @abstractmethod
    def action_space(self) -> List[ActionType]:
        raise NotImplementedError

    @abstractmethod
    def reset(self, config:Dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def render(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self, action:ActionType) -> StepInfoType:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> CloseInfoType:
        raise NotImplementedError