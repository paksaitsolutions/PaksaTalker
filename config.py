"""Configuration settings for PaksaTalker"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch

class PaksaConfig:
    """Configuration manager for PaksaTalker"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent.parent
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'paths': {
                'models': str(self.base_dir / 'models'),
                'output': str(self.base_dir / 'results'),
                'temp': str(self.base_dir / 'temp')
            },
            'models': {
                'sadtalker': {
                    'enabled': True,
                    'model_path': 'models/sadtalker',
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                },
                'wav2lip': {
                    'enabled': True,
                    'model_path': 'models/wav2lip',
                    'face_det_batch_size': 4,
                    'pads': [0, 10, 0, 0],
                    'resize_factor': 1
                },
                'gesture': {
                    'enabled': True,
                    'model_path': 'models/gesture',
                    'style': 'casual',
                    'intensity': 0.7
                },
                'qwen': {
                    'enabled': True,
                    'model_name': 'Qwen/Qwen2.5-Omni-7B',
                    'cache_dir': 'models/qwen',
                    'max_length': 500
                }
            },
            'video': {
                'fps': 25,
                'codec': 'mp4v',
                'resolution': [512, 512]
            },
            'logging': {
                'level': 'INFO',
                'file': 'paksatalker.log'
            }
        }
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
            self._update_config(self.config, user_config)
    
    def _update_config(self, base: Dict, update: Dict) -> Dict:
        """Recursively update configuration"""
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                base[key] = self._update_config(base[key], value)
            else:
                base[key] = value
        return base
    
    def save_config(self, path: str) -> None:
        """Save current configuration to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value

# Global configuration instance
config = PaksaConfig()
