"""
Breathing Simulation Module

This module provides functionality for simulating natural breathing patterns
in 3D characters, including both normal and emotional breathing variations.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class BreathingParameters:
    """Parameters controlling breathing behavior."""
    # Base breathing parameters
    base_rate: float = 0.2           # Base breathing rate (cycles per second)
    rate_variation: float = 0.1      # Random variation in breathing rate
    base_amplitude: float = 0.5      # Base breathing amplitude (0.0 to 1.0)
    amplitude_variation: float = 0.2  # Random variation in amplitude
    
    # Breathing pattern parameters
    inhale_ratio: float = 0.4        # Portion of breath cycle spent inhaling (0.0 to 1.0)
    hold_inhale_ratio: float = 0.1   # Portion of breath cycle to hold after inhale
    exhale_ratio: float = 0.4        # Portion of breath cycle spent exhaling
    hold_exhale_ratio: float = 0.1   # Portion of breath cycle to hold after exhale
    
    # Physical movement parameters
    chest_rise: float = 0.3          # How much the chest rises (0.0 to 1.0)
    shoulder_rise: float = 0.2       # How much the shoulders rise (0.0 to 1.0)
    head_bob: float = 0.1            # Subtle head movement (0.0 to 1.0)
    
    # Emotional state parameters
    emotional_state: str = "neutral"  # Current emotional state
    emotional_intensity: float = 0.0  # Intensity of emotional effect (0.0 to 1.0)
    
    # Randomness and variation
    random_seed: int = 42            # Random seed for reproducibility
    enable_variation: bool = True    # Whether to enable natural variation
    
    # Advanced parameters
    smoothness: float = 0.8          # How smooth the breathing is (0.0 to 1.0)
    recovery_rate: float = 0.5       # How quickly breathing returns to normal after variation

# Emotional state presets
EMOTIONAL_PRESETS = {
    "neutral": {
        "rate": 0.0,
        "amplitude": 0.0,
        "chest_rise": 0.0,
        "shoulder_rise": 0.0,
        "head_bob": 0.0
    },
    "calm": {
        "rate": -0.3,
        "amplitude": 0.3,
        "chest_rise": 0.2,
        "shoulder_rise": 0.1,
        "head_bob": 0.0
    },
    "excited": {
        "rate": 0.5,
        "amplitude": 0.4,
        "chest_rise": 0.4,
        "shoulder_rise": 0.3,
        "head_bob": 0.1
    },
    "nervous": {
        "rate": 0.7,
        "amplitude": 0.2,
        "chest_rise": 0.3,
        "shoulder_rise": 0.4,
        "head_bob": 0.2
    },
    "tired": {
        "rate": -0.4,
        "amplitude": 0.5,
        "chest_rise": 0.1,
        "shoulder_rise": 0.0,
        "head_bob": 0.3
    },
    "angry": {
        "rate": 0.6,
        "amplitude": 0.6,
        "chest_rise": 0.5,
        "shoulder_rise": 0.4,
        "head_bob": 0.1
    },
    "sad": {
        "rate": -0.2,
        "amplitude": 0.7,
        "chest_rise": 0.3,
        "shoulder_rise": -0.2,  # Slumped shoulders
        "head_bob": 0.4         # More pronounced head movement
    }
}

@dataclass
class BreathingState:
    """Current state of the breathing simulation."""
    phase: float = 0.0                  # Current phase in the breathing cycle (0 to 2π)
    time: float = 0.0                   # Current simulation time
    last_update: float = field(default_factory=lambda: time.time())  # Time of last update
    
    # Current parameters (with variations applied)
    current_rate: float = 0.0
    current_amplitude: float = 0.0
    
    # Target parameters (for smooth transitions)
    target_rate: float = 0.0
    target_amplitude: float = 0.0
    
    # Emotional state
    emotional_state: str = "neutral"
    emotional_intensity: float = 0.0
    
    # Random variations
    rate_variation: float = 0.0
    amplitude_variation: float = 0.0
    next_variation_time: float = 0.0
    
    # Breathing pattern
    inhale_ratio: float = 0.4
    hold_inhale_ratio: float = 0.1
    exhale_ratio: float = 0.4
    hold_exhale_ratio: float = 0.1

class BreathingSimulation:
    """
    Simulates natural breathing patterns for 3D characters.
    """
    
    def __init__(self, params: Optional[BreathingParameters] = None):
        """
        Initialize the breathing simulation.
        
        Args:
            params: Breathing parameters. If None, defaults will be used.
        """
        self.params = params if params is not None else BreathingParameters()
        self.state = BreathingState()
        
        # Set random seed for reproducibility
        random.seed(self.params.random_seed)
        np.random.seed(self.params.random_seed)
        
        # Initialize state
        self.state.current_rate = self.params.base_rate
        self.state.current_amplitude = self.params.base_amplitude
        self.state.target_rate = self.params.base_rate
        self.state.target_amplitude = self.params.base_amplitude
        self.state.emotional_state = self.params.emotional_state
        self.state.emotional_intensity = self.params.emotional_intensity
        
        # Set up breathing pattern
        self._update_breathing_pattern()
        
        # Initialize variation timer
        self._schedule_next_variation()
    
    def _update_breathing_pattern(self):
        """Update the breathing pattern based on current parameters."""
        # Normalize ratios to ensure they sum to 1.0
        total = (self.params.inhale_ratio + self.params.hold_inhale_ratio +
                self.params.exhale_ratio + self.params.hold_exhale_ratio)
        
        if total == 0:
            # Default pattern if all zeros
            self.state.inhale_ratio = 0.4
            self.state.hold_inhale_ratio = 0.1
            self.state.exhale_ratio = 0.4
            self.state.hold_exhale_ratio = 0.1
        else:
            # Normalize the ratios
            scale = 1.0 / total
            self.state.inhale_ratio = self.params.inhale_ratio * scale
            self.state.hold_inhale_ratio = self.params.hold_inhale_ratio * scale
            self.state.exhale_ratio = self.params.exhale_ratio * scale
            self.state.hold_exhale_ratio = self.params.hold_exhale_ratio * scale
    
    def _schedule_next_variation(self):
        """Schedule the next random variation in breathing."""
        if not self.params.enable_variation:
            self.state.rate_variation = 0.0
            self.state.amplitude_variation = 0.0
            return
            
        # Schedule next variation in 5-15 seconds
        self.state.next_variation_time = (
            time.time() + 
            random.uniform(5.0, 15.0) * (1.0 - self.params.smoothness * 0.9)
        )
        
        # Generate new variations
        self.state.rate_variation = random.uniform(
            -self.params.rate_variation,
            self.params.rate_variation
        )
        
        self.state.amplitude_variation = random.uniform(
            -self.params.amplitude_variation,
            self.params.amplitude_variation
        )
    
    def set_emotional_state(self, state: str, intensity: float = 1.0):
        """
        Set the emotional state of the breathing pattern.
        
        Args:
            state: Emotional state (e.g., 'calm', 'excited', 'nervous')
            intensity: Intensity of the emotional effect (0.0 to 1.0)
        """
        if state not in EMOTIONAL_PRESETS:
            print(f"Warning: Unknown emotional state '{state}'. Using 'neutral'.")
            state = "neutral"
            
        self.state.emotional_state = state
        self.state.emotional_intensity = max(0.0, min(1.0, intensity))
        
        # Update parameters based on emotional state
        preset = EMOTIONAL_PRESETS[state]
        
        # Apply emotional effects with intensity
        self.params.base_rate += preset["rate"] * intensity
        self.params.base_amplitude += preset["amplitude"] * intensity
        self.params.chest_rise += preset["chest_rise"] * intensity
        self.params.shoulder_rise += preset["shoulder_rise"] * intensity
        self.params.head_bob += preset["head_bob"] * intensity
        
        # Clamp values to valid ranges
        self.params.base_rate = max(0.01, min(2.0, self.params.base_rate))
        self.params.base_amplitude = max(0.0, min(1.0, self.params.base_amplitude))
        self.params.chest_rise = max(0.0, min(1.0, self.params.chest_rise))
        self.params.shoulder_rise = max(-1.0, min(1.0, self.params.shoulder_rise))
        self.params.head_bob = max(0.0, min(1.0, self.params.head_bob))
    
    def update(self, delta_time: Optional[float] = None):
        """
        Update the breathing simulation.
        
        Args:
            delta_time: Time since last update in seconds. If None, calculates automatically.
        """
        current_time = time.time()
        
        # Calculate delta time if not provided
        if delta_time is None:
            delta_time = current_time - self.state.last_update
            self.state.last_update = current_time
        
        # Check if we need to update variations
        if self.params.enable_variation and current_time >= self.state.next_variation_time:
            self._schedule_next_variation()
        
        # Update target values with variations
        self.state.target_rate = (
            self.params.base_rate * 
            (1.0 + self.state.rate_variation * self.params.smoothness)
        )
        
        self.state.target_amplitude = (
            self.params.base_amplitude * 
            (1.0 + self.state.amplitude_variation * self.params.smoothness)
        )
        
        # Smoothly interpolate towards target values
        recovery = min(1.0, delta_time * self.params.recovery_rate)
        self.state.current_rate += (
            (self.state.target_rate - self.state.current_rate) * recovery
        )
        
        self.state.current_amplitude += (
            (self.state.target_amplitude - self.state.current_amplitude) * recovery
        )
        
        # Update breathing phase
        phase_increment = 2.0 * math.pi * self.state.current_rate * delta_time
        self.state.phase = (self.state.phase + phase_increment) % (2.0 * math.pi)
        
        # Update time
        self.state.time += delta_time
    
    def get_breathing_value(self) -> float:
        """
        Get the current breathing value (0.0 to 1.0).
        
        Returns:
            Normalized breathing value where 0.0 is fully exhaled and 1.0 is fully inhaled.
        """
        # Map phase to [0, 1] range
        phase = self.state.phase / (2.0 * math.pi)
        
        # Calculate the breathing curve based on the current phase
        if phase < self.state.inhale_ratio:
            # Inhale phase
            t = phase / self.state.inhale_ratio
            return 0.5 * (1.0 - math.cos(t * math.pi))
        
        phase -= self.state.inhale_ratio
        if phase < self.state.hold_inhale_ratio:
            # Hold after inhale
            return 1.0
        
        phase -= self.state.hold_inhale_ratio
        if phase < self.state.exhale_ratio:
            # Exhale phase
            t = phase / self.state.exhale_ratio
            return 0.5 * (1.0 + math.cos(t * math.pi))
        
        # Hold after exhale
        return 0.0
    
    def get_chest_movement(self) -> float:
        """
        Get the current chest movement due to breathing.
        
        Returns:
            Vertical movement of the chest (normalized, 0.0 to 1.0).
        """
        breath = self.get_breathing_value()
        return breath * self.params.chest_rise * self.state.current_amplitude
    
    def get_shoulder_movement(self) -> float:
        """
        Get the current shoulder movement due to breathing.
        
        Returns:
            Vertical movement of the shoulders (normalized, -1.0 to 1.0).
        """
        breath = self.get_breathing_value()
        return breath * self.params.shoulder_rise * self.state.current_amplitude
    
    def get_head_movement(self) -> float:
        """
        Get the current head movement due to breathing.
        
        Returns:
            Vertical movement of the head (normalized, 0.0 to 1.0).
        """
        breath = self.get_breathing_value()
        return breath * self.params.head_bob * self.state.current_amplitude
    
    def get_breathing_state(self) -> Dict[str, float]:
        """
        Get the current state of the breathing simulation.
        
        Returns:
            Dictionary containing breathing parameters and state.
        """
        return {
            'phase': self.state.phase,
            'breath_value': self.get_breathing_value(),
            'chest_movement': self.get_chest_movement(),
            'shoulder_movement': self.get_shoulder_movement(),
            'head_movement': self.get_head_movement(),
            'rate': self.state.current_rate,
            'amplitude': self.state.current_amplitude,
            'emotional_state': self.state.emotional_state,
            'emotional_intensity': self.state.emotional_intensity,
            'time': self.state.time
        }
    
    def set_breathing_rate(self, rate: float):
        """
        Set the base breathing rate.
        
        Args:
            rate: Breathing rate in cycles per second (0.01 to 2.0)
        """
        self.params.base_rate = max(0.01, min(2.0, rate))
    
    def set_breathing_amplitude(self, amplitude: float):
        """
        Set the base breathing amplitude.
        
        Args:
            amplitude: Breathing amplitude (0.0 to 1.0)
        """
        self.params.base_amplitude = max(0.0, min(1.0, amplitude))
    
    def set_breathing_pattern(self, inhale: float, hold_inhale: float, 
                            exhale: float, hold_exhale: float):
        """
        Set the breathing pattern timing.
        
        Args:
            inhale: Time spent inhaling (relative units)
            hold_inhale: Time to hold after inhaling (relative units)
            exhale: Time spent exhaling (relative units)
            hold_exhale: Time to hold after exhaling (relative units)
        """
        self.params.inhale_ratio = max(0.0, inhale)
        self.params.hold_inhale_ratio = max(0.0, hold_inhale)
        self.params.exhale_ratio = max(0.0, exhale)
        self.params.hold_exhale_ratio = max(0.0, hold_exhale)
        self._update_breathing_pattern()
