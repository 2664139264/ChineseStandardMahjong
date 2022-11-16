import os
from collections import OrderedDict

import numpy as np
import torch

from typing import Dict, Union, List
from model_pool import ModelPool

class ChineseStandardMahjongModelPool(ModelPool):
    
    # model.state_dict() returns OrderedDict
    ModelType = OrderedDict
    ModelHandlerType = str
    DistributionType = Union[np.ndarray, str]
    
    _default_model_pool_path = './model_pool'
    _tag_version_seperator = '_'
    _model_file_name_format = '{tag}' + f'{_tag_version_seperator}' + '{version}.pkl'
    
    def __init__(self, config:Dict):
        self.config = config
        self._latest_version = 0
        self._file_names = list()
    
    @property
    def size(self) -> int:
        return self.config.get('size', 1)

    @property
    def path(self) -> str:
        return self.config.get('path', self._default_model_pool_path)
    
    def save_model(self, model:ModelType, tag:str='') -> ModelHandlerType:
        assert self._tag_version_seperator not in tag
        
        file_name = self._model_file_name_format.format(tag=tag, version=self._latest_version)
        
        path = os.path.join(self.path, file_name)

        torch.save(model, path)
        
        handler = self._latest_version
        self._file_names.append(file_name)
        self._latest_version += 1
        return handler
    
    def load_model(self, handler:ModelHandlerType) -> ModelType:
        path = os.path.join(self.path, self._model_file_name_format(tag=self._tags[handler], version=handler))
        model = torch.load(path)
        return model
    
    def sample_model(self, n:int, dist:DistributionType='latest') -> List[ModelHandlerType]:
        n_models = min(self.size, self._latest_version)
        assert n_models > 0
        
        if dist == 'uniform':
            dist = np.ones(n_models) / self.size
        elif dist == 'latest':
            dist = np.append(np.zeros(n_models-1), 1)
        
        start_version = self._latest_version - n_models
        end_version = self._latest_version
        
        return np.random.choice(range(start_version, end_version), size=n, p=dist).tolist()