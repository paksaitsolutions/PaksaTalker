"""
Micro-expressions module for subtle, involuntary facial expressions.

This module provides functionality for generating micro-expressions that
add realism to facial animations by simulating subtle, involuntary movements.
"""

import random
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

@dataclass
class MicroExpressionParameters:
    """Parameters controlling micro-expression behavior."""
    min_interval: float = 3.0         # Minimum time between micro-expressions (seconds)
    max_interval: float = 10.0        # Maximum time between micro-expressions (seconds)
    min_duration: float = 0.1         # Minimum duration of a micro-expression (seconds)
    max_duration: float = 0.5         # Maximum duration of a micro-expression (seconds)
    intensity: float = 0.3            # Overall intensity of micro-expressions (0.0 to 1.0)
    random_seed: int = 42             # Random seed for reproducibility

# Define common micro-expression types with their blend shape weights
MICRO_EXPRESSION_TYPES = {
    "brow_raise": {"brow_raiser_outer_l": 0.4, "brow_raiser_inner_l": 0.3},
    "brow_furrow": {"brow_lowerer_l": 0.3, "brow_lowerer_r": 0.3},
    "nose_scrunch": {"nose_wrinkler_l": 0.4, "nose_wrinkler_r": 0.4},
    "lip_corner_pull": {"lip_corner_puller_l": 0.3, "lip_corner_puller_r": 0.3},
    "lip_press": {"lip_presser_l": 0.4, "lip_presser_r": 0.4},
    "chin_raise": {"chin_raiser_b": 0.3},
    "lip_suck": {"lip_suck_l": 0.3, "lip_suck_r": 0.3},
    "jaw_drop": {"jaw_drop": 0.2}
}

@dataclass
class MicroExpressionState:
    """Current state of micro-expressions."""
    active: bool = False
    start_time: float = 0.0
    duration: float = 0.0
    expression_type: str = ""
    intensity: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    next_expression_time: float = 0.0

class MicroExpressionSystem:
    """
    Manages micro-expressions for facial animation.
    """
    
    def __init__(self, params: Optional[MicroExpressionParameters] = None):
        """
        Initialize the micro-expression system.
        
        Args:
            params: Micro-expression parameters. If None, defaults will be used.
        """
        self.params = params if params is not None else MicroExpressionParameters()
        self.state = MicroExpressionState()
        
        # Set random seed for reproducibility
        random.seed(self.params.random_seed)
        np.random.seed(self.params.random_seed)
        
        # Initialize next expression time
        self._initialize_next_expression()
    
    def _initialize_next_expression(self):
        """Set the time for the next micro-expression."""
        interval = random.uniform(
            self.params.min_interval,
            self.params.max_interval
        )
        self.state.next_expression_time = time.time() + interval
    
    def _select_random_expression(self) -> Tuple[str, Dict[str, float]]:
        """
        Randomly select a micro-expression type and its weights.
        
        Returns:
            Tuple of (expression_name, weights_dict)
        """
        expression_name = random.choice(list(MICRO_EXPRESSION_TYPES.keys()))
        base_weights = MICRO_EXPRESSION_TYPES[expression_name]
        
        # Apply some randomness to the weights
        weights = {}
        for k, v in base_weights.items():
            # Vary the weight by ±30%
            variation = random.uniform(0.7, 1.3)
            weights[k] = v * variation * self.params.intensity
        
        return expression_name, weights
    
    def update(self, current_time: Optional[float] = None):
        """
        Update the micro-expression state.
        
        Args:
            current_time: Current time in seconds. If None, uses time.time()
        """
        if current_time is None:
            current_time = time.time()
        
        if self.state.active:
            # Update active micro-expression
            elapsed = current_time - self.state.start_time
            
            if elapsed >= self.state.duration:
                # Micro-expression is complete
                self.state.active = False
                self._initialize_next_expression()
            else:
                # Calculate intensity curve (ease in-out)
                t = elapsed / self.state.duration
                if t < 0.5:
                    # Ease in
                    self.state.intensity = 2 * t * t
                else:
                    # Ease out
                    t = (t - 0.5) * 2
                    self.state.intensity = 1.0 - (1.0 - t) * (1.0 - t)
                
                # Apply intensity to weights
                for k in self.state.weights.keys():
                    self.state.weights[k] *= self.state.intensity
        
        # Check if it's time to start a new micro-expression
        elif current_time >= self.state.next_expression_time:
            self._trigger_expression()
    
    def _trigger_expression(self):
        """Trigger a new micro-expression."""
        self.state.active = True
        self.state.start_time = time.time()
        self.state.duration = random.uniform(
            self.params.min_duration,
            self.params.max_duration
        )
        
        # Select a random expression
        expr_type, weights = self._select_random_expression()
        self.state.expression_type = expr_type
        self.state.weights = weights
        self.state.intensity = 0.0  # Will be updated in update()
    
    def get_current_weights(self) -> Dict[str, float]:
        """
        Get the current blend shape weights from active micro-expressions.
        
        Returns:
            Dictionary of blend shape names to weights (0.0 to 1.0)
        """
        if not self.state.active:
            return {}
        
        # Scale weights by current intensity
        return {k: v * self.state.intensity for k, v in self.state.weights.items()}
    
    def get_state(self) -> Dict:
        """
        Get the current state of the micro-expression system.
        
        Returns:
            Dictionary containing state information
        """
        return {
            'active': self.state.active,
            'expression_type': self.state.expression_type if self.state.active else None,
            'intensity': self.state.intensity,
            'duration': self.state.duration if self.state.active else 0.0,
            'elapsed': (time.time() - self.state.start_time) if self.state.active else 0.0,
            'weights': self.state.weights if self.state.active else {}
        }
    
    def trigger_expression(self, expression_type: Optional[str] = None, 
                          duration: Optional[float] = None,
                          intensity: Optional[float] = None):
        """
        Manually trigger a specific micro-expression.
        
        Args:
            expression_type: Name of the expression to trigger. If None, a random one is chosen.
            duration: Duration of the expression in seconds. If None, a random duration is used.
            intensity: Intensity of the expression (0.0 to 1.0). If None, the default intensity is used.
        """
        self.state.active = True
        self.state.start_time = time.time()
        self.state.duration = duration if duration is not None else random.uniform(
            self.params.min_duration,
            self.params.max_duration
        )
        
        if expression_type is None or expression_type not in MICRO_EXPRESSION_TYPES:
            self.state.expression_type, self.state.weights = self._select_random_expression()
        else:
            self.state.expression_type = expression_type
            base_weights = MICRO_EXPRESSION_TYPES[expression_type]
            self.state.weights = {k: v * (intensity if intensity is not None else self.params.intensity) 
                               for k, v in base_weights.items()}
        
        self.state.intensity = 0.0  # Will be updated in update()
        self._initialize_next_expression()  # Reset the timer for the next automatic expression
