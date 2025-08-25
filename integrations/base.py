""Base integration class for PaksaTalker components"""
from abc import ABC, abstractmethod
import torch
from typing import Any, Dict, Optional

class BaseIntegration(ABC):
    """Abstract base class for all model integrations"""
    
    def __init__(self, device: str = None):
        """Initialize the integration
        
        Args:
            device: Device to run the model on ('cuda', 'mps', 'cpu')
        """
        self.device = device or self._get_default_device()
        self.model = None
        self.initialized = False
    
    def _get_default_device(self) -> str:
        """Get the default device based on availability"""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    @abstractmethod
    def load_model(self, *args, **kwargs) -> None:
        """Load the model and any required resources"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready"""
        return self.initialized and self.model is not None
    
    def to(self, device: str) -> None:
        """Move the model to the specified device"""
        if self.model and hasattr(self.model, 'to'):
            self.model = self.model.to(device)
            self.device = device
    
    def unload(self) -> None:
        """Unload the model and free resources"""
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            del self.model
            self.model = None
        self.initialized = False
    
    def __del__(self):
        """Cleanup on object deletion"""
        self.unload()
