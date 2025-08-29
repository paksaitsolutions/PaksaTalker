"""
Full Body Animation System for PaksaTalker
Handles lower body motion, weight shifting, foot placement, and balance physics.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    # Fallback rotation implementation
    class R:
        @staticmethod
        def from_euler(seq, angles, degrees=False):
            return FallbackRotation(angles)
        
        @staticmethod
        def from_quat(quat):
            return FallbackRotation(quat)
    
    class FallbackRotation:
        def __init__(self, data):
            self.data = np.atleast_1d(data)
        
        def as_quat(self):
            if len(self.data) == 4:
                return self.data
            # Simple euler to quaternion conversion
            return np.array([0, 0, 0, 1])
        
        def slerp(self, other, t):
            # Simple linear interpolation fallback
            return FallbackRotation(self.data * (1-t) + other.data * t)
        
        def __mul__(self, other):
            return FallbackRotation(self.data)

class StanceType(Enum):
    """Types of character stances."""
    NEUTRAL = auto()
    COMFORTABLE = auto()
    FORMAL = auto()
    CASUAL = auto()
    POWER = auto()
    RELAXED = auto()

class MovementType(Enum):
    """Types of character movements."""
    IDLE = auto()
    WALK = auto()
    TURN = auto()
    STEP = auto()
    SHIFT_WEIGHT = auto()

@dataclass
class FootState:
    """Represents the state of a single foot."""
    position: np.ndarray  # 3D position
    rotation: np.ndarray  # Quaternion rotation
    weight: float = 0.0   # 0.0 to 1.0, amount of weight on this foot
    is_planted: bool = True
    target_position: Optional[np.ndarray] = None
    target_rotation: Optional[np.ndarray] = None

@dataclass
class BalanceState:
    """Tracks the character's balance and center of mass."""
    com: np.ndarray  # Center of Mass (3D)
    com_velocity: np.ndarray = np.zeros(3)
    com_acceleration: np.ndarray = np.zeros(3)
    support_polygon: List[np.ndarray] = None  # Vertices of the support polygon
    balance_stability: float = 1.0  # 0.0 (falling) to 1.0 (perfect balance)
    is_recovering: bool = False
    recovery_timer: float = 0.0
    last_stable_com: np.ndarray = None  # Last known stable COM position
    last_stable_time: float = 0.0  # Time when last stable position was recorded

class FullBodyAnimator:
    """
    Handles full body animation including lower body movement, weight shifting,
    foot placement, and balance physics.
    """
    
    def __init__(self):
        # Initialize feet states (left and right)
        self.left_foot = FootState(
            position=np.array([-0.1, 0, 0]),
            rotation=np.array([0, 0, 0, 1]),  # Identity quaternion
            weight=0.5
        )
        self.right_foot = FootState(
            position=np.array([0.1, 0, 0]),
            rotation=np.array([0, 0, 0, 1]),
            weight=0.5
        )
        
        # Foot placement parameters
        self.foot_placement_radius = 0.8  # Max distance foot can be placed from body
        self.step_height = 0.15  # Maximum step height
        self.step_duration = 0.5  # Time for a complete step cycle (seconds)
        self.step_phase = 0.0  # Current phase in step cycle (0-1)
        self.stepping_foot = 'left'  # Which foot is currently stepping
        self.step_targets = {}  # Pre-computed step targets
        self.ground_heights = {}  # Terrain height at foot positions
        
        # Weight shifting parameters
        self.weight_shift_target = 0.5  # Target weight distribution (0.0 = all left, 1.0 = all right)
        self.weight_shift_velocity = 0.0  # Current weight shift velocity
        self.weight_shift_accel = 2.0  # How quickly weight shifts occur
        self.weight_shift_damping = 0.8  # Damping factor for weight shifting
        self.last_weight_shift_time = 0.0  # For timing weight shifts
        self.weight_shift_magnitude = 0.0  # Current magnitude of weight shift
        
        # Balance state
        self.balance = BalanceState(
            com=np.array([0, 0.9, 0]),  # Slightly above the pelvis
            support_polygon=self._calculate_support_polygon()
        )
        
        # Animation parameters
        self.stance = StanceType.NEUTRAL
        self.movement = MovementType.IDLE
        self.stance_width = 0.2  # Distance between feet
        self.stance_angle = 0.1  # Foot outward angle in radians
        
        # Physics parameters
        self.gravity = np.array([0, -9.81, 0])
        self.mass = 70.0  # kg
        self.com_height = 0.9  # meters
        
    def _calculate_support_polygon(self) -> List[np.ndarray]:
        """Calculate the current support polygon based on foot positions."""
        # Simple rectangular support polygon between feet
        # In a real implementation, this would consider foot shape and orientation
        return [
            self.left_foot.position + np.array([-0.1, 0, -0.1]),
            self.left_foot.position + np.array([0.1, 0, -0.1]),
            self.right_foot.position + np.array([0.1, 0, 0.1]),
            self.right_foot.position + np.array([-0.1, 0, 0.1])
        ]
    
    def update_weight_shifting(self, delta_time: float):
        """Update weight distribution between feet for natural movement."""
        # Calculate desired weight distribution based on movement and balance
        target_left_weight = 1.0 - self.weight_shift_target
        
        # Apply acceleration based on difference from target
        current_left_weight = self.left_foot.weight
        weight_diff = target_left_weight - current_left_weight
        
        # Update weight shift velocity with acceleration and damping
        self.weight_shift_velocity += weight_diff * self.weight_shift_accel * delta_time
        self.weight_shift_velocity *= self.weight_shift_damping
        
        # Update weights
        self.left_foot.weight += self.weight_shift_velocity * delta_time
        self.right_foot.weight = 1.0 - self.left_foot.weight
        
        # Update weight shift magnitude (for animation)
        self.weight_shift_magnitude = abs(self.left_foot.weight - 0.5) * 2.0
        
        # Ensure weights stay in valid range
        self.left_foot.weight = max(0.0, min(1.0, self.left_foot.weight))
        self.right_foot.weight = max(0.0, min(1.0, self.right_foot.weight))
    
    def shift_weight(self, amount: float, duration: float = 0.5):
        """
        Shift weight distribution between feet.
        
        Args:
            amount: -1.0 (full left) to 1.0 (full right)
            duration: Time to complete the weight shift in seconds
        """
        self.weight_shift_target = (amount + 1.0) / 2.0  # Convert to 0-1 range
        self.weight_shift_accel = 2.0 / max(0.1, duration)  # Adjust acceleration based on duration
    
    def _is_point_in_polygon(self, point: np.ndarray, polygon: List[np.ndarray]) -> bool:
        """Check if a 2D point is inside a polygon."""
        x, y = point[0], point[2]  # Project onto XZ plane
        n = len(polygon)
        inside = False
        
        p1x, p1z = polygon[0][0], polygon[0][2]
        for i in range(1, n + 1):
            p2x, p2z = polygon[i % n][0], polygon[i % n][2]
            if y > min(p1z, p2z):
                if y <= max(p1z, p2z):
                    if x <= max(p1x, p2x):
                        if p1z != p2z:
                            xinters = (y - p1z) * (p2x - p1x) / (p2z - p1z) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1z = p2x, p2z
            
        return inside
    
    def _calculate_stability(self, com: np.ndarray, cop: np.ndarray) -> float:
        """Calculate stability metric based on COM and COP positions."""
        # Project COM and COP to ground plane
        com_ground = com.copy()
        com_ground[1] = 0
        cop_ground = cop.copy()
        cop_ground[1] = 0
        
        # Distance from COM to COP
        distance = np.linalg.norm(com_ground - cop_ground)
        
        # Stability decreases with distance, but not linearly
        stability = 1.0 / (1.0 + distance * 2.0)
        
        # Check if COM is within support polygon
        if not self._is_point_in_polygon(com, self.balance.support_polygon):
            stability *= 0.5  # Penalize being outside support polygon
            
        return max(0.0, min(1.0, stability))
    
    def _apply_recovery_force(self, delta_time: float):
        """Apply recovery forces to regain balance."""
        if not self.balance.is_recovering:
            return
            
        # Calculate recovery direction (towards last stable COM)
        if self.balance.last_stable_com is None:
            self.balance.last_stable_com = self.balance.com.copy()
            
        recovery_dir = self.balance.last_stable_com - self.balance.com
        recovery_dist = np.linalg.norm(recovery_dir)
        
        if recovery_dist < 0.05:  # Close enough to stable position
            self.balance.is_recovering = False
            self.balance.recovery_timer = 0.0
            return
            
        # Normalize direction and apply recovery force
        if recovery_dist > 1e-6:
            recovery_dir = recovery_dir / recovery_dist
            recovery_force = recovery_dir * min(5.0, recovery_dist * 10.0)
            self.balance.com_velocity += recovery_force * delta_time
            
        # Update recovery timer
        self.balance.recovery_timer += delta_time
        if self.balance.recovery_timer > 2.0:  # Give up after 2 seconds
            self.balance.is_recovering = False
            self.balance.recovery_timer = 0.0
    
    def update_balance(self, delta_time: float):
        """Update balance state with physics-based recovery system."""
        # Update weight distribution
        self.update_weight_shifting(delta_time)
        
        # Calculate center of pressure (COP)
        total_weight = self.left_foot.weight + self.right_foot.weight
        if total_weight > 1e-6:
            cop = (
                self.left_foot.position * self.left_foot.weight +
                self.right_foot.position * self.right_foot.weight
            ) / total_weight
        else:
            cop = (self.left_foot.position + self.right_foot.position) * 0.5
        
        # Calculate center of mass (COM) acceleration
        com_to_cop = cop - self.balance.com
        com_accel = (self.gravity * com_to_cop[1] / self.com_height) + self.gravity
        
        # Apply damping to COM velocity
        self.balance.com_velocity *= 0.98  # Damping factor
        
        # Update COM state
        self.balance.com_velocity += com_accel * delta_time
        self.balance.com += self.balance.com_velocity * delta_time
        self.balance.com_acceleration = com_accel
        
        # Update support polygon
        self.balance.support_polygon = self._calculate_support_polygon()
        
        # Calculate stability
        prev_stability = self.balance.balance_stability
        self.balance.balance_stability = self._calculate_stability(
            self.balance.com, cop
        )
        
        # Check for balance loss
        if (self.balance.balance_stability < 0.3 and 
            not self.balance.is_recovering and 
            delta_time - self.balance.last_stable_time > 1.0):
            self.balance.is_recovering = True
            self.balance.recovery_timer = 0.0
            self.balance.last_stable_com = self.balance.com.copy()
        
        # Apply recovery if needed
        if self.balance.is_recovering:
            self._apply_recovery_force(delta_time)
            
        # Update last stable position if we're stable
        if self.balance.balance_stability > 0.7:
            self.balance.last_stable_com = self.balance.com.copy()
            self.balance.last_stable_time = delta_time
            
        # Limit COM height to prevent excessive bouncing
        if self.balance.com[1] < 0.7 * self.com_height:
            self.balance.com[1] = 0.7 * self.com_height
            self.balance.com_velocity[1] = max(0, self.balance.com_velocity[1])
            
        # Apply floor collision
        if self.balance.com[1] < 0.1:  # Simple ground plane
            self.balance.com[1] = 0.1
            self.balance.com_velocity[1] = 0.0
    
    def _calculate_foot_placement(self, foot: str, desired_velocity: np.ndarray) -> np.ndarray:
        """Calculate optimal foot placement based on movement direction and terrain."""
        # Base position is current foot position
        current_pos = getattr(self, f'{foot}_foot').position
        
        if np.linalg.norm(desired_velocity) < 0.1:
            # Standing still - maintain natural stance
            return current_pos
            
        # Calculate desired step direction
        direction = desired_velocity / (np.linalg.norm(desired_velocity) + 1e-6)
        
        # Calculate base target position
        if foot == 'left':
            stance_offset = np.array([-self.stance_width/2, 0, 0])
        else:
            stance_offset = np.array([self.stance_width/2, 0, 0])
            
        # Project desired movement onto foot's natural swing arc
        target_pos = current_pos + direction * 0.3  # Base step length
        target_pos += stance_offset
        
        # Limit step length based on natural range
        step_length = np.linalg.norm(target_pos - current_pos)
        max_step = self.foot_placement_radius * 0.8
        if step_length > max_step:
            target_pos = current_pos + (target_pos - current_pos) * (max_step / step_length)
            
        # Add some randomness to foot placement for natural look
        if step_length > 0.1:  # Only if actually moving
            target_pos += np.random.normal(0, 0.01, 3)  # Small random offset
            target_pos[1] = 0  # Keep on ground plane (terrain height would be added here)
            
        return target_pos
        
    def update_foot_placement(self, desired_velocity: np.ndarray, delta_time: float):
        """Update foot placement based on desired movement and weight distribution."""
        # Adjust movement based on weight shift
        if abs(self.weight_shift_magnitude) > 0.7:
            # If weight is significantly shifted, reduce movement speed
            desired_velocity = desired_velocity * (1.0 - (self.weight_shift_magnitude - 0.7) * 0.5)
            
        # Update step phase
        if self.movement == MovementType.WALK and np.linalg.norm(desired_velocity) > 0.1:
            # Calculate phase increment based on movement speed
            speed_factor = min(1.0, np.linalg.norm(desired_velocity) / 2.0)
            phase_increment = (delta_time / self.step_duration) * (0.5 + speed_factor * 0.5)
            self.step_phase = (self.step_phase + phase_increment) % 1.0
            
            # Switch stepping foot at midpoint of step
            if self.step_phase < 0.5 and hasattr(self, '_was_stepping') and self._was_stepping >= 0.5:
                self.stepping_foot = 'right' if self.stepping_foot == 'left' else 'left'
            self._was_stepping = self.step_phase
        else:
            # Reset step phase when not moving
            self.step_phase = 0.0
            
        # Calculate foot targets for both feet
        for foot in ['left', 'right']:
            if foot not in self.step_targets:
                self.step_targets[foot] = getattr(self, f'{foot}_foot').position.copy()
                
            if foot == self.stepping_foot and self.step_phase < 0.5:
                # Calculate new target for stepping foot
                self.step_targets[foot] = self._calculate_foot_placement(foot, desired_velocity)
                
                # Animate step
                foot_obj = getattr(self, f'{foot}_foot')
                if self.step_phase < 0.25:  # Lifting phase
                    t = self.step_phase * 4.0
                    height = np.sin(t * np.pi) * self.step_height
                    foot_obj.position[1] = height
                else:  # Lowering phase
                    t = (self.step_phase - 0.25) * 4.0
                    height = np.sin((1.0 - t) * np.pi) * self.step_height * 0.5
                    # Linear interpolation (lerp)
                    t = min(1.0, delta_time * 10.0)
                    foot_obj.position = foot_obj.position * (1 - t) + self.step_targets[foot] * t
                    foot_obj.position[1] = height
            else:
                # Non-stepping foot maintains position
                foot_obj = getattr(self, f'{foot}_foot')
                # Linear interpolation (lerp)
                t = min(1.0, delta_time * 5.0)
                foot_obj.position = foot_obj.position * (1 - t) + self.step_targets[foot] * t
                
        # Update foot rotations based on movement direction
        if np.linalg.norm(desired_velocity) > 0.1:
            direction = desired_velocity / (np.linalg.norm(desired_velocity) + 1e-6)
            target_rot = R.from_euler('y', np.arctan2(direction[0], direction[2])).as_quat()
            
            for foot in ['left', 'right']:
                foot_obj = getattr(self, f'{foot}_foot')
                # Only rotate if foot is planted or just about to lift
                if foot != self.stepping_foot or self.step_phase > 0.4:
                    foot_obj.rotation = R.from_quat(foot_obj.rotation).slerp(
                        R.from_quat(target_rot),
                        min(1.0, delta_time * 5.0)
                    ).as_quat()
        
        # Update movement state
        if np.linalg.norm(desired_velocity) > 0.1:
            self.movement = MovementType.WALK
        else:
            self.movement = MovementType.IDLE
            step_length = 0.6  # meters
            step_height = 0.1  # meters
            
            # Alternate foot steps based on movement direction
            if desired_velocity[2] > 0.1:  # Moving forward
                if not hasattr(self, '_step_phase'):
                    self._step_phase = 0.0
                    self._stepping_foot = 'left'
                
                self._step_phase = min(1.0, self._step_phase + delta_time * 2.0)
                
                if self._step_phase >= 1.0:
                    self._step_phase = 0.0
                    self._stepping_foot = 'right' if self._stepping_foot == 'left' else 'left'
                
                # Animate stepping
                if self._stepping_foot == 'left':
                    self._animate_step(self.left_foot, step_length, step_height, self._step_phase)
                else:
                    self._animate_step(self.right_foot, step_length, step_height, self._step_phase)
    
    def _animate_step(self, foot: FootState, length: float, height: float, phase: float):
        """Animate a single step for a foot."""
        if phase < 0.5:  # Lifting phase
            t = phase * 2.0
            foot.position[1] = t * height
        else:  # Lowering phase
            t = (phase - 0.5) * 2.0
            foot.position[1] = (1.0 - t) * height
            foot.position[2] += length * 0.1  # Small forward movement
    
    def set_stance(self, stance_type: StanceType):
        """Set the character's stance and adjust weight distribution accordingly."""
        self.stance = stance_type
        
        # Adjust stance parameters based on type
        if stance_type == StanceType.FORMAL:
            self.stance_width = 0.15
            self.stance_angle = 0.0
            self.shift_weight(0.0, 0.3)  # Centered weight for formal stance
        elif stance_type == StanceType.CASUAL:
            self.stance_width = 0.25
            self.stance_angle = 0.15
            self.shift_weight(0.1, 0.5)  # Slight weight shift for casual stance
        elif stance_type == StanceType.POWER:
            self.stance_width = 0.3
            self.stance_angle = 0.2
            self.shift_weight(0.0, 0.4)  # Strong centered stance
        elif stance_type == StanceType.RELAXED:
            self.stance_width = 0.35
            self.stance_angle = 0.25
            self.shift_weight(0.3, 0.7)  # Shift more weight to one side for relaxed stance
    
    def _apply_stance(self):
        """Apply the current stance to the character's pose with balance compensation."""
        # In a real implementation, this would adjust the character's pose
        # to match the desired stance while maintaining balance
        
        # Apply balance compensation to stance
        if self.balance.is_recovering:
            # During recovery, adjust stance to help regain balance
            recovery_lean = (self.balance.last_stable_com - self.balance.com) * 0.5
            # Apply recovery lean to both feet
            self.left_foot.position[0] -= recovery_lean[0] * 0.5
            self.left_foot.position[2] -= recovery_lean[2] * 0.5
            self.right_foot.position[0] -= recovery_lean[0] * 0.5
            self.right_foot.position[2] -= recovery_lean[2] * 0.5
    
    def update(self, delta_time: float, desired_velocity: np.ndarray = None):
        """Update the full body animation system with balance recovery."""
        if desired_velocity is None:
            desired_velocity = np.zeros(3)
        
        # Scale desired velocity based on current stability
        stability_factor = min(1.0, self.balance.balance_stability * 1.5)
        desired_velocity = desired_velocity * stability_factor
        
        # Update balance and physics first
        self.update_balance(delta_time)
        
        # Update foot placement based on movement and current balance
        self.update_foot_placement(desired_velocity, delta_time)
        
        # Apply stance settings with balance compensation
        self._apply_stance()
        
        # Apply additional stabilization if balance is critical
        if self.balance.balance_stability < 0.4:
            # Slow down movement when balance is poor
            self.movement = MovementType.IDLE
            
            # Automatically adjust foot positions for better balance
            com_ground = self.balance.com.copy()
            com_ground[1] = 0
            
            for foot in [self.left_foot, self.right_foot]:
                to_com = com_ground - foot.position
                to_com[1] = 0  # Only horizontal adjustment
                dist = np.linalg.norm(to_com)
                if dist > 0.5:  # If foot is too far from COM
                    foot.position += to_com * 0.1  # Move foot toward COM
    
    def _apply_stance(self):
        """Apply the current stance to the character's pose."""
        # In a real implementation, this would adjust the character's pose
        # to match the desired stance while maintaining balance
        pass

# Example usage
if __name__ == "__main__":
    animator = FullBodyAnimator()
    animator.set_stance(StanceType.NEUTRAL)
    
    # Simulate walking forward
    animator.movement = MovementType.WALK
    desired_velocity = np.array([0, 0, 1.0])  # Move forward at 1 m/s
    
    # Simulate 2 seconds of animation at 60 FPS
    for i in range(120):
        animator.update(1.0/60.0, desired_velocity)
        # In a real application, you would update the character's skeleton here
