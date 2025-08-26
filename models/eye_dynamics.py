"""
Eye and Face Dynamics Module

This module provides functionality for realistic eye and facial movements,
including blinking, eye saccades, and micro-expressions.
"""

import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

@dataclass
class SaccadeParameters:
    """Parameters controlling saccade behavior."""
    min_saccade_interval: float = 1.0  # Minimum time between saccades (seconds)
    max_saccade_interval: float = 4.0  # Maximum time between saccades (seconds)
    saccade_duration: float = 0.05     # Duration of a saccade (seconds)
    max_saccade_angle: float = 20.0    # Maximum saccade angle in degrees
    saccade_speed: float = 15.0        # Speed of saccade movement
    random_seed: int = 42              # Random seed for reproducibility

@dataclass
class BlinkParameters:
    """Parameters controlling blink behavior."""
    min_blink_interval: float = 2.0  # Minimum time between blinks (seconds)
    max_blink_interval: float = 8.0  # Maximum time between blinks (seconds)
    blink_duration: float = 0.2      # Duration of a blink (seconds)
    blink_speed: float = 5.0         # Speed of blink (higher = faster)
    blink_intensity: float = 1.0     # Intensity of blink (0.0 to 1.0)
    random_seed: int = 42            # Random seed for reproducibility

@dataclass
class EyeState:
    """Current state of the eyes."""
    # Blink state
    is_blinking: bool = False
    blink_start_time: float = 0.0
    blink_progress: float = 0.0
    next_blink_time: float = 0.0
    
    # Eye openness (0.0 = fully closed, 1.0 = fully open)
    left_eye_open: float = 1.0
    right_eye_open: float = 1.0
    
    # Gaze direction (normalized coordinates, 0,0 is center)
    look_target: Tuple[float, float] = (0, 0)
    target_look_target: Tuple[float, float] = (0, 0)
    
    # Saccade state
    is_saccading: bool = False
    saccade_start_time: float = 0.0
    saccade_progress: float = 0.0
    next_saccade_time: float = 0.0
    saccade_start_pos: Tuple[float, float] = (0, 0)

class EyeDynamics:
    """
    Manages realistic eye movements including blinking and saccades.
    """
    
    def __init__(self, blink_params: Optional[BlinkParameters] = None, 
                 saccade_params: Optional[SaccadeParameters] = None):
        """
        Initialize the eye dynamics system.
        
        Args:
            blink_params: Blink parameters. If None, defaults will be used.
            saccade_params: Saccade parameters. If None, defaults will be used.
        """
        self.blink_params = blink_params if blink_params is not None else BlinkParameters()
        self.saccade_params = saccade_params if saccade_params is not None else SaccadeParameters()
        
        # Initialize state
        self.state = EyeState()
        self._last_update_time = time.time()
        
        # Set random seeds for reproducibility
        seed = self.blink_params.random_seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize timers
        self._initialize_next_blink()
        self._initialize_next_saccade()
        
        # Initialize look target
        self.state.look_target = (0, 0)
        self.state.target_look_target = (0, 0)
    
    def _initialize_next_blink(self):
        """Set the time for the next blink."""
        interval = random.uniform(
            self.blink_params.min_blink_interval,
            self.blink_params.max_blink_interval
        )
        self.state.next_blink_time = time.time() + interval
    
    def _initialize_next_saccade(self):
        """Set the time for the next saccade."""
        interval = random.uniform(
            self.saccade_params.min_saccade_interval,
            self.saccade_params.max_saccade_interval
        )
        self.state.next_saccade_time = time.time() + interval
    
    def _get_random_saccade_target(self) -> Tuple[float, float]:
        """Generate a random saccade target within the allowed range."""
        # Generate random angle and distance
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0.2, 0.8)  # Keep within a reasonable range
        
        # Convert to cartesian coordinates
        x = math.cos(angle) * distance
        y = math.sin(angle) * distance * 0.7  # Flatten vertically (eyes move more horizontally)
        
        return (x, y)
    
    def _update_saccade(self, current_time: float):
        """Update saccade state based on current time."""
        if self.state.is_saccading:
            # Update saccade progress
            elapsed = current_time - self.state.saccade_start_time
            self.state.saccade_progress = min(1.0, elapsed / self.saccade_params.saccade_duration)
            
            # Apply smooth step for saccade movement
            t = self.state.saccade_progress
            t_smooth = t * t * (3 - 2 * t)  # Smooth step function
            
            # Interpolate between start and target positions
            start_x, start_y = self.state.saccade_start_pos
            target_x, target_y = self.state.target_look_target
            
            current_x = start_x + (target_x - start_x) * t_smooth
            current_y = start_y + (target_y - start_y) * t_smooth
            
            self.state.look_target = (current_x, current_y)
            
            # Check if saccade is complete
            if self.state.saccade_progress >= 1.0:
                self.state.is_saccading = False
                self.state.look_target = self.state.target_look_target
                self._initialize_next_saccade()
        
        # Check if it's time to start a new saccade
        elif current_time >= self.state.next_saccade_time and not self.state.is_blinking:
            self.state.is_saccading = True
            self.state.saccade_start_time = current_time
            self.state.saccade_progress = 0.0
            self.state.saccade_start_pos = self.state.look_target
            self.state.target_look_target = self._get_random_saccade_target()
    
    def _update_blink(self, current_time: float):
        """Update blink state based on current time."""
        if self.state.is_blinking:
            # Update blink progress
            elapsed = current_time - self.state.blink_start_time
            self.state.blink_progress = min(1.0, elapsed / self.blink_params.blink_duration)
            
            # Calculate eye openness using a smooth step function
            t = self.state.blink_progress
            if t < 0.5:
                # Closing eyes (first half of blink)
                open_amount = 1.0 - (t * 2) ** self.blink_params.blink_speed
            else:
                # Opening eyes (second half of blink)
                open_amount = ((t - 0.5) * 2) ** (1.0 / self.blink_params.blink_speed)
            
            # Apply some natural asymmetry to the blink
            self.state.left_eye_open = np.clip(open_amount + random.uniform(-0.1, 0.1), 0, 1)
            self.state.right_eye_open = np.clip(open_amount + random.uniform(-0.1, 0.1), 0, 1)
            
            # Check if blink is complete
            if self.state.blink_progress >= 1.0:
                self.state.is_blinking = False
                self.state.left_eye_open = 1.0
                self.state.right_eye_open = 1.0
                self._initialize_next_blink()
        
        # Check if it's time to start a new blink
        elif current_time >= self.state.next_blink_time:
            self.state.is_blinking = True
            self.state.blink_start_time = current_time
            self.state.blink_progress = 0.0
    
    def update(self):
        """
        Update the eye state based on the current time.
        Call this once per frame.
        """
        current_time = time.time()
        self._update_blink(current_time)
        self._update_saccade(current_time)
        self._last_update_time = current_time
    
    def look_at(self, target: Tuple[float, float], immediate: bool = False):
        """
        Make the eyes look at a specific target.
        
        Args:
            target: (x, y) coordinates to look at (normalized -1 to 1)
            immediate: If True, jump to the target immediately. If False, use saccade.
        """
        if immediate:
            self.state.look_target = target
            self.state.target_look_target = target
            self.state.is_saccading = False
            self._initialize_next_saccade()
        else:
            self.state.target_look_target = target
            if not self.state.is_saccading and not self.state.is_blinking:
                # Start a new saccade to the target
                self.state.is_saccading = True
                self.state.saccade_start_time = time.time()
                self.state.saccade_progress = 0.0
                self.state.saccade_start_pos = self.state.look_target
    
    def get_eye_states(self) -> Dict[str, float]:
        """
        Get the current state of the eyes.
        
        Returns:
            Dictionary containing eye state information:
            - left_eye_open: 0.0 (closed) to 1.0 (open)
            - right_eye_open: 0.0 (closed) to 1.0 (open)
            - look_x, look_y: Current gaze direction (normalized)
            - target_look_x, target_look_y: Target gaze direction (normalized)
            - is_blinking: boolean indicating if a blink is in progress
            - is_saccading: boolean indicating if a saccade is in progress
        """
        return {
            'left_eye_open': self.state.left_eye_open,
            'right_eye_open': self.state.right_eye_open,
            'look_x': self.state.look_target[0],
            'look_y': self.state.look_target[1],
            'target_look_x': self.state.target_look_target[0],
            'target_look_y': self.state.target_look_target[1],
            'is_blinking': self.state.is_blinking,
            'is_saccading': self.state.is_saccading
        }
    
    def force_blink(self, duration: Optional[float] = None):
        """
        Force the character to blink.
        
        Args:
            duration: Optional custom duration for the blink. If None, uses the default.
        """
        self.state.is_blinking = True
        self.state.blink_start_time = time.time()
        self.state.blink_progress = 0.0
        if duration is not None:
            self.blink_params.blink_duration = duration
    
    def force_saccade(self, target: Optional[Tuple[float, float]] = None):
        """
        Force the character to perform a saccade to a specific target.
        
        Args:
            target: Optional (x, y) target to look at. If None, a random target is chosen.
        """
        if target is not None:
            self.state.target_look_target = target
        
        if not self.state.is_blinking:  # Don't start a saccade during a blink
            self.state.is_saccading = True
            self.state.saccade_start_time = time.time()
            self.state.saccade_progress = 0.0
            self.state.saccade_start_pos = self.state.look_target
            
            # If no target was provided, generate a random one
            if target is None:
                self.state.target_look_target = self._get_random_saccade_target()
