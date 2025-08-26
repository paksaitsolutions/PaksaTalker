"""Configuration management for PaksaTalker."""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dotenv import load_dotenv

# Try to import torch, but don't fail if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class Config:
    """Configuration manager for PaksaTalker."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Path to configuration YAML file. If None, uses default paths.
        """
        # Load environment variables from .env file if it exists
        load_dotenv()

        # Default configuration
        self._config = {
            'app': {
                'name': 'PaksaTalker',
                'version': '0.1.0',
                'debug': os.getenv('DEBUG', 'False').lower() == 'true',
            },
            'paths': {
                'root': str(Path(__file__).parent.parent),
                'models': os.getenv('MODELS_DIR', 'models'),
                'output': os.getenv('OUTPUT_DIR', 'output'),
                'temp': os.getenv('TEMP_DIR', 'temp'),
            },
            'models': {
                'sadtalker': {
                    'enabled': True,
                    'model_path': os.getenv('SADTALKER_MODEL_PATH', 'models/sadtalker'),
                    'device': os.getenv('DEVICE', 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
                },
                'wav2lip': {
                    'enabled': True,
                    'model_path': os.getenv('WAV2LIP_MODEL_PATH', 'models/wav2lip'),
                    'device': os.getenv('DEVICE', 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
                },
                'gesture': {
                    'enabled': True,
                    'model_path': os.getenv('GESTURE_MODEL_PATH', 'models/gesture'),
                    'device': os.getenv('DEVICE', 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
                },
                'qwen': {
                    'enabled': True,
                    'model_name': os.getenv('QWEN_MODEL_NAME', 'Qwen/Qwen2.5-Omni-7B'),
                    'device': os.getenv('DEVICE', 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
                },
            },
            'api': {
                'host': os.getenv('API_HOST', '0.0.0.0'),
                'port': int(os.getenv('API_PORT', '8000')),
                'workers': int(os.getenv('API_WORKERS', '1')),
                'debug': os.getenv('API_DEBUG', 'False').lower() == 'true',
            },
        }

        # Update with YAML config if provided
        if config_path and os.path.exists(config_path):
            self._update_from_yaml(config_path)

        # Ensure all directories exist
        self._ensure_directories()

    def _update_from_yaml(self, config_path: str) -> None:
        """Update configuration from a YAML file.

        Args:
            config_path: Path to YAML configuration file.
        """
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
                self._deep_update(self._config, yaml_config)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")

    def _deep_update(self, original: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively update a dictionary.

        Args:
            original: Dictionary to update.
            update: Dictionary with updates.
        """
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        paths = self._config['paths']

        # Make paths absolute if they are relative
        for key in ['models', 'output', 'temp']:
            path = paths[key]
            if not os.path.isabs(path):
                paths[key] = os.path.join(paths['root'], path)

        # Create directories if they don't exist
        for path in paths.values():
            if path != paths['root']:  # Skip root directory
                os.makedirs(path, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot notation.

        Args:
            key: Dot-notation key (e.g., 'models.sadtalker.enabled')
            default: Default value if key is not found

        Returns:
            The configuration value or default if not found
        """
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using bracket notation."""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()

    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the configuration. If None, saves to 'config/config.yaml'.
        """
        if path is None:
            path = os.path.join(self['paths.root'], 'config', 'config.yaml')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            yaml.safe_dump(self._config, f, default_flow_style=False, sort_keys=False)

# Create default config instance
config = Config()

__all__ = ['Config', 'config']
