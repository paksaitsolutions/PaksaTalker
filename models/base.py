'''Base model class for PaksaTalker models.'''
import os
import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BaseModel(ABC):
    '''Abstract base class for all PaksaTalker models.'''

    def __init__(self, device: Optional[str] = None):
        '''Initialize the model.

        Args:
            device: Device to run the model on ('cuda', 'mps', 'cpu').
            If None, automatically selects the best available device.
        '''
        self.device = device or self._get_available_device()
        self.model = None
        self.initialized = False

    def _get_available_device(self) -> str:
        '''Get the best available device.'''
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    @abstractmethod
    def load_model(self, **kwargs) -> None:
        '''Load the model weights and initialize the model.'''
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        '''Check if the model is loaded.'''
        return self.initialized and self.model is not None

    def unload(self) -> None:
        '''Unload the model and free up memory.'''
        if self.model is not None:
            if hasattr(self.model, 'to'):
                self.model.to('cpu')
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.initialized = False

    def __del__(self):
        '''Cleanup on object deletion.'''
        self.unload()

    def to(self, device: str) -> None:
        '''Move model to the specified device.'''
        if self.model is not None and hasattr(self.model, 'to'):
            self.model.to(device)
            self.device = device

    def eval(self) -> None:
        '''Set the model to evaluation mode.'''
        if self.model is not None and hasattr(self.model, 'eval'):
            self.model.eval()

    def train(self, mode: bool = True) -> None:
        '''Set the model to training mode.'''
        if self.model is not None and hasattr(self.model, 'train'):
            self.model.train(mode)

    def state_dict(self) -> Dict[str, Any]:
        '''Get the model's state dictionary.'''
        if self.model is not None and hasattr(self.model, 'state_dict'):
            return self.model.state_dict()
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        '''Load the model's state dictionary.'''
        if self.model is not None and hasattr(self.model, 'load_state_dict'):
            self.model.load_state_dict(state_dict, strict=strict)
