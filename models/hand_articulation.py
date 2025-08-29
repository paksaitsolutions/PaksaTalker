"""
Hand Articulation System for PaksaTalker
Handles finger movements, gestures, and interactions.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
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

class HandSide(Enum):
    LEFT = auto()
    RIGHT = auto()

class FingerType(Enum):
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4

@dataclass
class FingerJoint:
    """Represents a single joint in a finger."""
    rotation: np.ndarray  # Quaternion rotation
    bend: float = 0.0  # 0.0 (straight) to 1.0 (fully bent)
    twist: float = 0.0  # -1.0 to 1.0 for rotation around bone axis

@dataclass
class Finger:
    """Represents a finger with multiple joints."""
    joints: List[FingerJoint]
    length: float
    base_rotation: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0])
    )
    current_pose: Dict[str, float] = field(default_factory=dict)

@dataclass
class HandState:
    """Represents the complete state of a hand."""
    side: HandSide
    wrist_rotation: np.ndarray  # Quaternion
    wrist_position: np.ndarray  # 3D position
    fingers: Dict[FingerType, Finger]
    is_grabbing: bool = False
    grab_strength: float = 0.0  # 0.0 (open) to 1.0 (fully closed)
    current_gesture: str = "relaxed"

class HandArticulator:
    """Manages hand articulation, gestures, and transitions."""
    
    def __init__(self, side: HandSide = HandSide.RIGHT):
        self.side = side
        self.hand = self._create_default_hand()
        
        # Transition settings
        self.gesture_transition_speed = 3.0  # Base transition speed
        self.finger_curl_speed = 4.0  # Base curl speed
        self.transition_easing = self._ease_in_out_quad  # Default easing function
        self.transition_progress = 0.0
        self.transition_time = 0.0
        self.current_blend = 0.0  # Current blend amount between poses (0-1)
        
        # Gesture blending
        self.current_gesture = "relaxed"
        self.next_gesture = "relaxed"
        self.blend_gesture = None
        self.blend_start_time = 0.0
        self.blend_duration = 0.3  # Default blend duration in seconds
        
        # Predefined gestures
        self.gestures = {
            "relaxed": self._create_relaxed_pose,
            "fist": self._create_fist_pose,
            "point": self._create_point_pose,
            "pinch": self._create_pinch_pose,
            "thumbs_up": self._create_thumbs_up_pose,
            "open_hand": self._create_open_hand_pose,
            "peace": self._create_peace_pose,
            "rock": self._create_rock_pose,
            "ok": self._create_ok_pose,
        }
        
        # Initialize with relaxed pose
        self._apply_pose_to_fingers(self.gestures["relaxed"]())
    
    def _create_default_hand(self) -> HandState:
        """Create a hand with default joint configuration."""
        fingers = {}
        finger_lengths = {
            FingerType.THUMB: 0.1,
            FingerType.INDEX: 0.12,
            FingerType.MIDDLE: 0.13,
            FingerType.RING: 0.12,
            FingerType.PINKY: 0.1
        }
        
        for finger_type in FingerType:
            # 3 joints per finger (proximal, intermediate, distal)
            joints = [FingerJoint(np.array([0.0, 0.0, 0.0, 1.0])) for _ in range(3)]
            fingers[finger_type] = Finger(
                joints=joints,
                length=finger_lengths[finger_type]
            )
        
        return HandState(
            side=self.side,
            wrist_rotation=np.array([0.0, 0.0, 0.0, 1.0]),
            wrist_position=np.zeros(3),
            fingers=fingers
        )
    
    def _create_relaxed_pose(self) -> Dict[str, float]:
        """Return the relaxed hand pose configuration."""
        return {
            "thumb_bend": 0.3,
            "index_bend": 0.2,
            "middle_bend": 0.1,
            "ring_bend": 0.2,
            "pinky_bend": 0.3,
            "spread": 0.0
        }
    
    def _create_fist_pose(self) -> Dict[str, float]:
        """Return the fist hand pose configuration."""
        return {
            "thumb_bend": 0.8,
            "index_bend": 1.0,
            "middle_bend": 1.0,
            "ring_bend": 1.0,
            "pinky_bend": 1.0,
            "spread": 0.0
        }
    
    def _create_point_pose(self) -> Dict[str, float]:
        """Return the pointing hand pose configuration."""
        return {
            "thumb_bend": 0.5,
            "index_bend": 0.1,
            "middle_bend": 0.9,
            "ring_bend": 0.9,
            "pinky_bend": 0.9,
            "spread": 0.1
        }
    
    def _create_pinch_pose(self) -> Dict[str, float]:
        """Return the pinch hand pose configuration."""
        return {
            "thumb_bend": 0.7,
            "index_bend": 0.7,
            "middle_bend": 0.9,
            "ring_bend": 0.9,
            "pinky_bend": 0.9,
            "spread": 0.1
        }
    
    def _create_thumbs_up_pose(self) -> Dict[str, float]:
        """Return the thumbs up hand pose configuration."""
        return {
            "thumb_bend": 0.1,
            "index_bend": 0.9,
            "middle_bend": 0.9,
            "ring_bend": 0.9,
            "pinky_bend": 0.9,
            "spread": 0.0
        }
    
    def _create_open_hand_pose(self) -> Dict[str, float]:
        """Return the open hand pose configuration."""
        return {
            "thumb_bend": 0.1,
            "index_bend": 0.0,
            "middle_bend": 0.0,
            "ring_bend": 0.0,
            "pinky_bend": 0.1,
            "spread": 0.0,
            "thumb_spread": 0.0,
            "pinky_spread": 0.0
        }
        
    def _create_peace_pose(self) -> Dict[str, float]:
        """Return the peace sign pose configuration."""
        return {
            "thumb_bend": 0.5,
            "index_bend": 0.0,
            "middle_bend": 0.0,
            "ring_bend": 1.0,
            "pinky_bend": 1.0,
            "spread": 0.2,
            "thumb_spread": 0.3,
            "pinky_spread": -0.1
        }
        
    def _create_rock_pose(self) -> Dict[str, float]:
        """Return the rock sign pose configuration."""
        return {
            "thumb_bend": 0.8,
            "index_bend": 0.1,
            "middle_bend": 1.0,
            "ring_bend": 1.0,
            "pinky_bend": 0.1,
            "spread": 0.3,
            "thumb_spread": 0.4,
            "pinky_spread": -0.2
        }
        
    def _create_ok_pose(self) -> Dict[str, float]:
        """Return the OK sign pose configuration."""
        return {
            "thumb_bend": 0.3,
            "index_bend": 0.3,
            "middle_bend": 0.0,
            "ring_bend": 0.0,
            "pinky_bend": 0.0,
            "spread": 0.1,
            "thumb_spread": 0.5,
            "pinky_spread": 0.0,
            "thumb_tip_touch": 1.0  # Special parameter for thumb-index pinch
        }
    
    def set_gesture(self, gesture_name: str, transition_time: float = 0.3, 
                   easing: str = "in_out_quad"):
        """
        Transition to a new hand gesture.
        
        Args:
            gesture_name: Name of the gesture to transition to
            transition_time: Duration of the transition in seconds
            easing: Easing function to use for the transition. Options:
                   'linear', 'ease_in', 'ease_out', 'ease_in_out', 'bounce'
        """
        if gesture_name not in self.gestures:
            raise ValueError(f"Unknown gesture: {gesture_name}")
            
        if gesture_name == self.current_gesture:
            return  # Already in the target gesture
            
        # Set up transition
        self.next_gesture = gesture_name
        self.blend_start_time = 0.0
        self.blend_duration = max(0.01, transition_time)
        self.current_blend = 0.0
        
        # Set easing function
        easing_functions = {
            "linear": self._ease_linear,
            "ease_in": self._ease_in_quad,
            "ease_out": self._ease_out_quad,
            "ease_in_out": self._ease_in_out_quad,
            "bounce": self._ease_out_bounce,
        }
        self.transition_easing = easing_functions.get(easing, self._ease_in_out_quad)
        
        # Store current pose for blending
        self.current_gesture = gesture_name
        self.hand.current_gesture = gesture_name
    
    def update(self, delta_time: float):
        """
        Update hand state and animations.
        
        Args:
            delta_time: Time in seconds since last update
        """
        # Update gesture blending
        if self.blend_start_time < self.blend_duration:
            self.blend_start_time += delta_time
            self.current_blend = min(1.0, self.blend_start_time / self.blend_duration)
            
            # Get current and target poses
            current_pose = self.gestures[self.current_gesture]()
            target_pose = self.gestures[self.next_gesture]()
            
            # Apply easing to the blend factor
            t = self.transition_easing(self.current_blend)
            
            # Create blended pose
            blended_pose = {}
            all_keys = set(current_pose.keys()).union(target_pose.keys())
            for key in all_keys:
                current_val = current_pose.get(key, 0.0)
                target_val = target_pose.get(key, 0.0)
                blended_pose[key] = current_val * (1 - t) + target_val * t
            
            # Apply the blended pose
            self._apply_pose_to_fingers(blended_pose)
    
    # Easing functions for smooth transitions
    @staticmethod
    def _ease_linear(t: float) -> float:
        return t
        
    @staticmethod
    def _ease_in_quad(t: float) -> float:
        return t * t
        
    @staticmethod
    def _ease_out_quad(t: float) -> float:
        return 1 - (1 - t) * (1 - t)
        
    @staticmethod
    def _ease_in_out_quad(t: float) -> float:
        return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
        
    @staticmethod
    def _ease_out_bounce(t: float) -> float:
        n1 = 7.5625
        d1 = 2.75
        
        if t < 1 / d1:
            return n1 * t * t
        elif t < 2 / d1:
            t -= 1.5 / d1
            return n1 * t * t + 0.75
        elif t < 2.5 / d1:
            t -= 2.25 / d1
            return n1 * t * t + 0.9375
        else:
            t -= 2.625 / d1
            return n1 * t * t + 0.984375
    
    def _apply_pose_to_fingers(self, pose: Dict[str, float]):
        """
        Apply a pose configuration to all fingers with advanced controls.
        
        Args:
            pose: Dictionary containing pose parameters
        """
        # Map pose parameters to finger bends and spreads
        finger_params = {
            FingerType.THUMB: {
                'bend': pose.get("thumb_bend", 0.0),
                'spread': pose.get("thumb_spread", 0.0),
                'twist': pose.get("thumb_twist", 0.0)
            },
            FingerType.INDEX: {
                'bend': pose.get("index_bend", 0.0),
                'spread': pose.get("index_spread", 0.0)
            },
            FingerType.MIDDLE: {
                'bend': pose.get("middle_bend", 0.0),
                'spread': pose.get("middle_spread", 0.0)
            },
            FingerType.RING: {
                'bend': pose.get("ring_bend", 0.0),
                'spread': pose.get("ring_spread", 0.0)
            },
            FingerType.PINKY: {
                'bend': pose.get("pinky_bend", 0.0),
                'spread': pose.get("pinky_spread", 0.0)
            }
        }
        
        # Apply parameters to each finger
        for finger_type, finger in self.hand.fingers.items():
            params = finger_params[finger_type]
            bend_amount = params['bend']
            spread = params.get('spread', 0.0)
            
            # Apply bend to each joint with natural variation
            for i, joint in enumerate(finger.joints):
                # Base bend with natural falloff
                joint_bend = bend_amount * (1.0 - (i * 0.3))
                
                # Add some natural curl variation
                if i > 0:
                    joint_bend *= 0.9 + (0.1 * (i / len(finger.joints)))
                
                # Apply spread to the base joint
                if i == 0 and 'spread' in params:
                    # Convert spread to rotation (in radians)
                    spread_rad = spread * 0.3  # Scale factor for spread
                    if finger_type == FingerType.THUMB:
                        # Thumb has different spread direction
                        spread_rad *= -1.0 if self.side == HandSide.RIGHT else 1.0
                    joint.rotation = R.from_euler('y', spread_rad).as_quat()
                
                # Apply twist if specified (mostly for thumb)
                if 'twist' in params and i == 0:
                    twist_rad = params['twist'] * 0.5  # Scale factor for twist
                    twist_rot = R.from_euler('x', twist_rad).as_quat()
                    joint.rotation = R.from_quat(joint.rotation) * R.from_quat(twist_rot)
                
                # Apply the final bend
                joint.bend = min(1.0, max(0.0, joint_bend))
            
            # Store current pose for reference
            finger.current_pose = pose
            
        # Special case: thumb-index pinch
        if pose.get("thumb_tip_touch", 0.0) > 0.0:
            self._apply_thumb_index_pinch(pose["thumb_tip_touch"])
    
    def _apply_thumb_index_pinch(self, strength: float):
        """
        Special handling for thumb-index pinch gesture.
        
        Args:
            strength: 0.0 (no pinch) to 1.0 (full pinch)
        """
        if strength <= 0.0:
            return
            
        # Get thumb and index fingers
        thumb = self.hand.fingers[FingerType.THUMB]
        index = self.hand.fingers[FingerType.INDEX]
        
        # Calculate target positions for pinch
        thumb_tip = self.get_finger_tip_position(FingerType.THUMB)
        index_tip = self.get_finger_tip_position(FingerType.INDEX)
        
        # Calculate midpoint between tips
        midpoint = (thumb_tip + index_tip) * 0.5
        
        # Apply pinch by adjusting the last joint of thumb and index
        if len(thumb.joints) > 0:
            thumb.joints[-1].bend = strength
        if len(index.joints) > 0:
            index.joints[-1].bend = strength
            
        # Slight wrist rotation for more natural pinch
        if strength > 0.5:
            rot_axis = np.array([0, 1, 0]) if self.side == HandSide.RIGHT else np.array([0, -1, 0])
            rot_amount = 0.2 * (strength - 0.5) * 2.0  # Only apply when strength > 0.5
            self.hand.wrist_rotation = R.from_rotvec(rot_axis * rot_amount).as_quat()
    
    def get_finger_tip_position(self, finger_type: FingerType) -> np.ndarray:
        """Calculate the 3D position of a finger tip."""
        finger = self.hand.fingers[finger_type]
        pos = self.hand.wrist_position.copy()
        
        # Apply finger base offset based on hand side and finger type
        side = 1.0 if self.side == HandSide.RIGHT else -1.0
        finger_offsets = {
            FingerType.THUMB: np.array([0.05 * side, 0.0, 0.02]),
            FingerType.INDEX: np.array([0.02 * side, 0.0, 0.1]),
            FingerType.MIDDLE: np.array([0.0, 0.0, 0.12]),
            FingerType.RING: np.array([-0.02 * side, 0.0, 0.1]),
            FingerType.PINKY: np.array([-0.04 * side, 0.0, 0.08])
        }
        
        pos += finger_offsets[finger_type]
        
        # Apply finger joints
        joint_length = finger.length / len(finger.joints)
        for joint in finger.joints:
            # In a real implementation, this would apply the joint rotations
            # For now, just move forward by joint length
            pos[2] += joint_length
            
        return pos
    
    def set_grab_strength(self, strength: float):
        """Set how strongly the hand is grabbing (0.0 to 1.0)."""
        self.hand.grab_strength = max(0.0, min(1.0, strength))
        
        # Adjust finger curls based on grab strength
        for finger_type, finger in self.hand.fingers.items():
            for joint in finger.joints:
                # More curl for stronger grab
                target_bend = self.hand.grab_strength
                joint.bend = target_bend
    
    def add_custom_gesture(self, name: str, pose_creator):
        """Add a custom hand gesture."""
        self.gestures[name] = pose_creator

# Example usage with advanced transitions
if __name__ == "__main__":
    import time
    
    # Create a hand
    hand = HandArticulator(HandSide.RIGHT)
    
    # Define some gestures to cycle through
    gestures = [
        ("open_hand", 1.0, "ease_in_out"),
        ("fist", 0.5, "bounce"),
        ("point", 0.7, "ease_out"),
        ("peace", 0.8, "ease_in"),
        ("rock", 0.6, "ease_in_out"),
        ("ok", 0.5, "bounce")
    ]
    
    # Animation loop
    for gesture_name, duration, easing in gestures:
        print(f"Transitioning to {gesture_name} with {easing} easing")
        hand.set_gesture(gesture_name, duration, easing)
        
        # Animate the transition
        frames = int(duration * 10)  # 10 updates per second
        for _ in range(frames):
            hand.update(0.1)  # 100ms per frame
            time.sleep(0.1)  # Simulate real-time
    
    # Test thumb-index pinch
    print("Testing thumb-index pinch")
    hand.set_gesture("open_hand")
    hand.update(1.0)  # Ensure we're in the open hand pose
    
    # Animate pinch
    for i in range(10):
        hand._apply_thumb_index_pinch(i / 9.0)  # 0.0 to 1.0
        time.sleep(0.1)
