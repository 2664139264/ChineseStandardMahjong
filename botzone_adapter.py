from typing import Callable
from abc import ABCMeta, abstractmethod  

from agent import Agent

class BotzoneAdapter(metaclass=ABCMeta):

    # load from request_stream: Iterable
    # output response using agent: Agent
    @abstractmethod
    def load_botzone_request_and_generate_response(self, agent:Agent, request_loader:Callable=input) -> None:
        raise NotImplementedError
