"""
Asymmetrical Expressions Module

This module provides functionality for creating natural asymmetrical facial expressions
that mimic the subtle asymmetries found in human faces.
"""

import random
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time

@dataclass
class AsymmetryParameters:
    """Parameters controlling asymmetrical expression behavior."""
    intensity: float = 0.3            # Base intensity of asymmetries (0.0 to 1.0)
    min_delay: float = 0.5           # Minimum time between asymmetry changes (seconds)
    max_delay: float = 3.0           # Maximum time between asymmetry changes (seconds)
    min_duration: float = 0.5        # Minimum duration of an asymmetry (seconds)
    max_duration: float = 2.0        # Maximum duration of an asymmetry (seconds)
    random_seed: int = 42            # Random seed for reproducibility
    enabled: bool = True             # Whether asymmetrical expressions are enabled

# Define common facial regions and their asymmetrical variations
FACIAL_REGIONS = {
    # Eyebrows
    'brow_outer_l': {'min': -0.3, 'max': 0.0, 'smoothness': 0.8},
    'brow_outer_r': {'min': -0.3, 'max': 0.0, 'smoothness': 0.8},
    'brow_inner_l': {'min': -0.2, 'max': 0.2, 'smoothness': 0.7},
    'brow_inner_r': {'min': -0.2, 'max': 0.2, 'smoothness': 0.7},
    
    # Eyes
    'eye_open_l': {'min': -0.1, 'max': 0.1, 'smoothness': 0.9},
    'eye_open_r': {'min': -0.1, 'max': 0.1, 'smoothness': 0.9},
    'eye_wide_l': {'min': -0.2, 'max': 0.0, 'smoothness': 0.8},
    'eye_wide_r': {'min': -0.2, 'max': 0.0, 'smoothness': 0.8},
    
    # Mouth
    'smile_l': {'min': -0.2, 'max': 0.2, 'smoothness': 0.7},
    'smile_r': {'min': -0.2, 'max': 0.2, 'smoothness': 0.7},
    'mouth_open': {'min': -0.15, 'max': 0.15, 'smoothness': 0.8},
    'mouth_wide': {'min': -0.1, 'max': 0.1, 'smoothness': 0.9},
    
    # Nose
    'nose_wrinkle_l': {'min': 0.0, 'max': 0.3, 'smoothness': 0.6},
    'nose_wrinkle_r': {'min': 0.0, 'max': 0.3, 'smoothness': 0.6},
    
    # Cheeks
    'cheek_raise_l': {'min': 0.0, 'max': 0.2, 'smoothness': 0.7},
    'cheek_raise_r': {'min': 0.0, 'max': 0.2, 'smoothness': 0.7},
}

@dataclass
class AsymmetryState:
    """Current state of asymmetrical expressions."""
    active_asymmetries: Dict[str, float] = field(default_factory=dict)
    target_asymmetries: Dict[str, float] = field(default_factory=dict)
    start_times: Dict[str, float] = field(default_factory=dict)
    end_times: Dict[str, float] = field(default_factory=dict)
    next_update_time: float = 0.0

class AsymmetricalExpressionSystem:
    """
    Manages asymmetrical facial expressions to add natural variation.
    """
    
    def __init__(self, params: Optional[AsymmetryParameters] = None):
        """
        Initialize the asymmetrical expression system.
        
        Args:
            params: Asymmetry parameters. If None, defaults will be used.
        """
        self.params = params if params is not None else AsymmetryParameters()
        self.state = AsymmetryState()
        
        # Set random seed for reproducibility
        random.seed(self.params.random_seed)
        np.random.seed(self.params.random_seed)
        
        # Initialize first update
        self._schedule_next_update()
    
    def _schedule_next_update(self):
        """Schedule the next update time."""
        delay = random.uniform(self.params.min_delay, self.params.max_delay)
        self.state.next_update_time = time.time() + delay
    
    def _generate_asymmetry_value(self, region: str) -> float:
        """Generate a random asymmetry value for a region."""
        params = FACIAL_REGIONS[region]
        value = random.uniform(params['min'], params['max'])
        return value * self.params.intensity
    
    def _smoothstep(self, t: float) -> float:
        """Smoothstep function for smooth transitions."""
        t = max(0, min(1, t))  # Clamp to [0, 1]
        return t * t * (3 - 2 * t)
    
    def update(self, current_time: Optional[float] = None):
        """
        Update the asymmetrical expressions.
        
        Args:
            current_time: Current time in seconds. If None, uses time.time()
        """
        if not self.params.enabled:
            return
            
        if current_time is None:
            current_time = time.time()
        
        # Check if it's time to update any asymmetries
        if current_time >= self.state.next_update_time:
            self._update_asymmetries(current_time)
            self._schedule_next_update()
        
        # Update active asymmetries with smooth transitions
        self._update_transitions(current_time)
    
    def _update_asymmetries(self, current_time: float):
        """Update which asymmetries are active."""
        # Randomly select a subset of regions to modify
        num_changes = random.randint(1, 3)
        regions = random.sample(list(FACIAL_REGIONS.keys()), num_changes)
        
        for region in regions:
            if region in self.state.active_asymmetries:
                # Keep some existing asymmetries
                if random.random() < 0.3:  # 30% chance to keep existing asymmetry
                    continue
            
            # Generate new target asymmetry
            self.state.target_asymmetries[region] = self._generate_asymmetry_value(region)
            self.state.start_times[region] = current_time
            self.state.end_times[region] = current_time + random.uniform(
                self.params.min_duration, self.params.max_duration
            )
    
    def _update_transitions(self, current_time: float):
        """Update active asymmetries with smooth transitions."""
        regions_to_remove = []
        
        for region, start_time in list(self.state.start_times.items()):
            end_time = self.state.end_times.get(region, current_time)
            
            if current_time >= end_time:
                # Transition complete
                if region in self.state.target_asymmetries:
                    # Set final value and remove from active transitions
                    self.state.active_asymmetries[region] = self.state.target_asymmetries[region]
                    del self.state.target_asymmetries[region]
                    del self.state.start_times[region]
                    del self.state.end_times[region]
                    
                    # Randomly decide if we should remove this asymmetry
                    if random.random() < 0.5:  # 50% chance to remove the asymmetry
                        regions_to_remove.append(region)
            else:
                # Calculate interpolation factor
                duration = end_time - start_time
                elapsed = current_time - start_time
                t = self._smoothstep(elapsed / duration)
                
                # Interpolate between current and target value
                current_val = self.state.active_asymmetries.get(region, 0.0)
                target_val = self.state.target_asymmetries.get(region, 0.0)
                
                # Apply smooth transition
                self.state.active_asymmetries[region] = current_val + (target_val - current_val) * t
        
        # Clean up completed asymmetries
        for region in regions_to_remove:
            if region in self.state.active_asymmetries:
                del self.state.active_asymmetries[region]
    
    def get_asymmetries(self) -> Dict[str, float]:
        """
        Get the current asymmetrical expression weights.
        
        Returns:
            Dictionary mapping region names to asymmetry values (-1.0 to 1.0)
        """
        return self.state.active_asymmetries.copy()
    
    def set_enabled(self, enabled: bool):
        """Enable or disable asymmetrical expressions."""
        self.params.enabled = enabled
        if not enabled:
            # Clear all active asymmetries when disabling
            self.state.active_asymmetries.clear()
            self.state.target_asymmetries.clear()
            self.state.start_times.clear()
            self.state.end_times.clear()
    
    def force_asymmetry(self, region: str, value: float, duration: float = 1.0):
        """
        Force a specific asymmetry value for a region.
        
        Args:
            region: Name of the facial region
            value: Asymmetry value (-1.0 to 1.0)
            duration: How long to maintain this asymmetry (seconds)
        """
        if region not in FACIAL_REGIONS:
            raise ValueError(f"Unknown facial region: {region}")
            
        current_time = time.time()
        self.state.target_asymmetries[region] = value
        self.state.start_times[region] = current_time
        self.state.end_times[region] = current_time + duration
        
        # Ensure the region is in active_asymmetries
        if region not in self.state.active_asymmetries:
            self.state.active_asymmetries[region] = 0.0
