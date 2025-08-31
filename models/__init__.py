"""PaksaTalker models package (lazy exports).

This package provides convenient, lazy-loaded accessors for core and
integrated models so importing `models` does not eagerly pull heavy deps.

Examples:
    from models import SadTalkerModel, Wav2LipModel
    from models import get_emage_model, FusionEngine
    from models import get_wav2lip2_model, QwenModel
"""

from typing import TYPE_CHECKING

# For type checkers only (no runtime cost)
if TYPE_CHECKING:
    from .base import BaseModel  # noqa: F401
    from .sadtalker import SadTalkerModel  # noqa: F401
    from .sadtalker_full import SadTalkerFull  # noqa: F401
    from .wav2lip import Wav2LipModel  # noqa: F401
    from .wav2lip2_aoti import Wav2Lip2AOTI, get_wav2lip2_model  # noqa: F401
    from .gesture import GestureModel  # noqa: F401
    from .qwen import QwenModel  # noqa: F401
    from .qwen_omni import QwenOmniModel, get_qwen_model  # noqa: F401
    from .optimization import (
        ModelOptimizer, optimize_model_for_device, benchmark_model  # noqa: F401
    )
    from .emage_realistic import EMageRealistic, get_emage_model  # noqa: F401
    from .fusion.engine import FusionEngine  # noqa: F401
    from .expression.engine import (
        ExpressionResult, detect_capabilities, estimate_from_path  # noqa: F401
    )

__all__ = [
    # Base + core wrappers
    'BaseModel',
    'SadTalkerModel',
    'SadTalkerFull',
    'Wav2LipModel',
    'Wav2Lip2AOTI', 'get_wav2lip2_model',
    'GestureModel',
    'QwenModel',
    'QwenOmniModel', 'get_qwen_model',
    # Optimizer utils
    'ModelOptimizer', 'optimize_model_for_device', 'benchmark_model',
    # EMAGE
    'EMageRealistic', 'get_emage_model',
    # Fusion
    'FusionEngine',
    # Expression analysis helpers
    'ExpressionResult', 'detect_capabilities', 'estimate_from_path',
]


def __getattr__(name: str):
    """Lazy attribute loader for submodules to avoid heavy imports at package import time."""
    if name == 'BaseModel':
        from .base import BaseModel
        return BaseModel
    if name == 'SadTalkerModel':
        from .sadtalker import SadTalkerModel
        return SadTalkerModel
    if name == 'SadTalkerFull':
        from .sadtalker_full import SadTalkerFull
        return SadTalkerFull
    if name == 'Wav2LipModel':
        from .wav2lip import Wav2LipModel
        return Wav2LipModel
    if name in ('Wav2Lip2AOTI', 'get_wav2lip2_model'):
        from .wav2lip2_aoti import Wav2Lip2AOTI, get_wav2lip2_model
        return {'Wav2Lip2AOTI': Wav2Lip2AOTI, 'get_wav2lip2_model': get_wav2lip2_model}[name]
    if name == 'GestureModel':
        from .gesture import GestureModel
        return GestureModel
    if name == 'QwenModel':
        from .qwen import QwenModel
        return QwenModel
    if name in ('QwenOmniModel', 'get_qwen_model'):
        from .qwen_omni import QwenOmniModel, get_qwen_model
        return {'QwenOmniModel': QwenOmniModel, 'get_qwen_model': get_qwen_model}[name]
    if name in ('ModelOptimizer', 'optimize_model_for_device', 'benchmark_model'):
        from .optimization import ModelOptimizer, optimize_model_for_device, benchmark_model
        return {
            'ModelOptimizer': ModelOptimizer,
            'optimize_model_for_device': optimize_model_for_device,
            'benchmark_model': benchmark_model,
        }[name]
    if name in ('EMageRealistic', 'get_emage_model'):
        from .emage_realistic import EMageRealistic, get_emage_model
        return {'EMageRealistic': EMageRealistic, 'get_emage_model': get_emage_model}[name]
    if name == 'FusionEngine':
        from .fusion.engine import FusionEngine
        return FusionEngine
    if name in ('ExpressionResult', 'detect_capabilities', 'estimate_from_path'):
        from .expression.engine import ExpressionResult, detect_capabilities, estimate_from_path
        return {
            'ExpressionResult': ExpressionResult,
            'detect_capabilities': detect_capabilities,
            'estimate_from_path': estimate_from_path,
        }[name]
    raise AttributeError(name)
