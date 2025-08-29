"""
Emotion-Based Gestures Module

This module provides functionality for generating gestures based on emotional context,
including emotion-gesture mapping and intensity modulation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import random
import numpy as np
from enum import Enum, auto

class EmotionType(Enum):
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    ANGRY = auto()
    SURPRISED = auto()
    DISGUSTED = auto()
    FEARFUL = auto()
    EXCITED = auto()
    CONFUSED = auto()
    THINKING = auto()
    CONFIDENT = auto()
    DISAGREEING = auto()
    AGREEING = auto()

class CulturalContext(Enum):
    WESTERN = auto()
    EAST_ASIAN = auto()
    MIDDLE_EASTERN = auto()
    SOUTH_ASIAN = auto()
    LATIN_AMERICAN = auto()
    AFRICAN = auto()
    GLOBAL = auto()  # Default neutral style

@dataclass
class GestureParameters:
    """Parameters for gesture generation based on emotion and context."""
    # Base parameters
    intensity: float = 0.5      # Base intensity (0.0 to 1.0)
    speed: float = 1.0          # Base speed multiplier
    smoothness: float = 0.7     # How smooth the transitions are (0.0 to 1.0)
    cultural_context: CulturalContext = CulturalContext.GLOBAL
    
    # Contextual parameters
    formality: float = 0.5      # 0.0 (casual) to 1.0 (formal)
    engagement: float = 0.7     # 0.0 (disengaged) to 1.0 (highly engaged)
    dominance: float = 0.5      # 0.0 (submissive) to 1.0 (dominant)
    
    # Emotion-specific parameters
    emotion_intensity: Dict[EmotionType, float] = field(default_factory=dict)
    emotion_speed: Dict[EmotionType, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # Default emotion parameters
        if not self.emotion_intensity:
            self.emotion_intensity = {
                EmotionType.NEUTRAL: 0.3,
                EmotionType.HAPPY: 0.8,
                EmotionType.SAD: 0.4,
                EmotionType.ANGRY: 0.9,
                EmotionType.SURPRISED: 0.7,
                EmotionType.DISGUSTED: 0.6,
                EmotionType.FEARFUL: 0.7
            }
        
        if not self.emotion_speed:
            self.emotion_speed = {
                EmotionType.NEUTRAL: 1.0,
                EmotionType.HAPPY: 1.2,
                EmotionType.SAD: 0.8,
                EmotionType.ANGRY: 1.3,
                EmotionType.SURPRISED: 1.5,
                EmotionType.DISGUSTED: 1.0,
                EmotionType.FEARFUL: 0.9
            }

class GestureType(Enum):
    # Basic gestures
    POINT = auto()
    NOD = auto()
    SHAKE_HEAD = auto()
    SHRUG = auto()
    WAVE = auto()
    WAVE_AWAY = auto()
    
    # Emotional gestures
    CLENCH_FIST = auto()  # Anger
    FACE_PALM = auto()    # Disappointment
    TOUCH_CHEST = auto()  # Sincerity
    SPREAD_ARMS = auto()  # Happiness, openness
    CROSS_ARMS = auto()   # Defensive, closed-off
    RUB_HANDS = auto()    # Anticipation, nervousness
    TOUCH_FACE = auto()   # Thoughtful, considering
    
    # Conversational gestures
    ENUMERATE = auto()    # Counting points
    OPEN_PALM = auto()    # Honesty, openness
    CHOPPING = auto()     # Emphasis
    
    # Enhanced emotional expressions
    HANDS_AT_REST = auto()      # Neutral, relaxed
    HAND_TO_CHEST = auto()      # Sincerity, emphasis
    HEAD_TILT = auto()          # Curiosity, interest
    HAND_ON_HEART = auto()      # Sincerity, gratitude
    HEAD_DOWN = auto()          # Shyness, submission
    SLOW_NOD = auto()           # Understanding, agreement
    EYEBROWS_RAISED = auto()    # Surprise, interest
    OPEN_PALM_UP = auto()       # Openness, offering
    HEAD_BACK = auto()          # Contemplation, skepticism
    NOSE_WRINKLE = auto()       # Disgust, distaste
    ARMS_CLOSE = auto()         # Insecurity, cold
    STEP_BACK = auto()          # Surprise, fear
    BIG_GESTURE = auto()        # Excitement, emphasis
    BOUNCY = auto()             # Energy, enthusiasm
    QUICK_NOD = auto()          # Eagerness, agreement
    CHIN_STROKE = auto()        # Thoughtfulness, evaluation
    FINGER_TO_LIP = auto()      # Contemplation, shushing
    SMALL_SMILE = auto()        # Politeness, subtle happiness
    LIGHT_NOD = auto()          # Acknowledgment, listening
    TIGHT_LIPS = auto()         # Disapproval, tension
    SLOW_BLINK = auto()         # Calmness, thoughtfulness
    EMPHATIC_HAND = auto()      # Emphasis, strong feeling
    
    # Special gestures
    PAUSE = auto()              # Natural pause in speech

@dataclass
class GestureDefinition:
    """Definition of a gesture with its parameters."""
    gesture_type: GestureType
    intensity: float = 1.0
    speed: float = 1.0
    symmetry: float = 1.0  # 1.0 = fully symmetrical, 0.0 = fully asymmetrical
    duration: float = 1.0  # In seconds
    
    def apply_emotion(self, emotion: EmotionType, intensity: float = 1.0):
        """Apply emotion-based modifications to the gesture."""
        # Get emotion parameters
        emotion_params = {
            EmotionType.NEUTRAL: (1.0, 1.0),
            EmotionType.HAPPY: (1.2, 1.3),
            EmotionType.SAD: (0.7, 0.8),
            EmotionType.ANGRY: (1.4, 1.2),
            EmotionType.SURPRISED: (1.1, 1.5),
            EmotionType.DISGUSTED: (1.0, 0.9),
            EmotionType.FEARFUL: (0.8, 1.1)
        }
        
        intensity_scale, speed_scale = emotion_params.get(emotion, (1.0, 1.0))
        
        # Apply intensity modulation
        self.intensity = min(1.0, self.intensity * intensity_scale * intensity)
        self.speed = self.speed * speed_scale
        
        # Adjust duration based on speed
        self.duration = max(0.5, min(2.0, self.duration / speed_scale))
        
        return self

class EmotionGestureMapper:
    """Maps emotions to gestures with intensity modulation and cultural adaptation."""
    
    def __init__(self, params: Optional[GestureParameters] = None):
        self.params = params if params is not None else GestureParameters()
        self._setup_gesture_mappings()
        self._setup_cultural_adaptations()
        self._context = {
            'previous_gestures': [],
            'conversation_topic': None,
            'interlocutor_style': None,
            'environment': 'neutral'  # 'formal', 'casual', 'professional', 'social'
        }
    
    def _setup_gesture_mappings(self):
        """Initialize the default emotion to gesture mappings with cultural variations."""
        self._gesture_mappings = {
            CulturalContext.GLOBAL: {
                EmotionType.NEUTRAL: [
                    (GestureType.NOD, 0.4),
                    (GestureType.OPEN_PALM, 0.3),
                    (GestureType.HANDS_AT_REST, 0.3)
                ],
                EmotionType.HAPPY: [
                    (GestureType.OPEN_PALM, 0.4),
                    (GestureType.HAND_TO_CHEST, 0.3),
                    (GestureType.HEAD_TILT, 0.3)
                ],
                EmotionType.SAD: [
                    (GestureType.HAND_ON_HEART, 0.5),
                    (GestureType.HEAD_DOWN, 0.3),
                    (GestureType.SLOW_NOD, 0.2)
                ],
                EmotionType.ANGRY: [
                    (GestureType.POINT, 0.4),
                    (GestureType.CHOPPING, 0.4),
                    (GestureType.CLENCH_FIST, 0.2)
                ],
                EmotionType.SURPRISED: [
                    (GestureType.EYEBROWS_RAISED, 0.5),
                    (GestureType.OPEN_PALM_UP, 0.3),
                    (GestureType.HEAD_BACK, 0.2)
                ],
                EmotionType.DISGUSTED: [
                    (GestureType.NOSE_WRINKLE, 0.5),
                    (GestureType.WAVE_AWAY, 0.3),
                    (GestureType.SHAKE_HEAD, 0.2)
                ],
                EmotionType.FEARFUL: [
                    (GestureType.ARMS_CLOSE, 0.5),
                    (GestureType.HEAD_DOWN, 0.3),
                    (GestureType.STEP_BACK, 0.2)
                ],
                EmotionType.EXCITED: [
                    (GestureType.BIG_GESTURE, 0.5),
                    (GestureType.BOUNCY, 0.3),
                    (GestureType.QUICK_NOD, 0.2)
                ],
                EmotionType.THINKING: [
                    (GestureType.CHIN_STROKE, 0.4),
                    (GestureType.HEAD_TILT, 0.3),
                    (GestureType.FINGER_TO_LIP, 0.3)
                ]
            },
            CulturalContext.EAST_ASIAN: {
                # More restrained gestures, less arm movement, more subtle expressions
                EmotionType.HAPPY: [
                    (GestureType.SMALL_SMILE, 0.6),
                    (GestureType.LIGHT_NOD, 0.4)
                ],
                EmotionType.ANGRY: [
                    (GestureType.TIGHT_LIPS, 0.6),
                    (GestureType.SLOW_BLINK, 0.4)
                ]
            },
            CulturalContext.MIDDLE_EASTERN: {
                # More expressive hand gestures, closer proximity
                EmotionType.HAPPY: [
                    (GestureType.HAND_ON_HEART, 0.5),
                    (GestureType.EMPHATIC_HAND, 0.5)
                ],
                EmotionType.THINKING: [
                    (GestureType.CHIN_STROKE, 0.6),
                    (GestureType.FINGER_TO_LIP, 0.4)
                ],
                EmotionType.AGREEING: [
                    (GestureType.NOD, 0.7),
                    (GestureType.OPEN_PALM, 0.3)
                ]
            },
            CulturalContext.SOUTH_ASIAN: {
                # Head movements, respectful gestures
                EmotionType.HAPPY: [
                    (GestureType.HEAD_TILT, 0.6),
                    (GestureType.SMALL_SMILE, 0.4)
                ],
                EmotionType.AGREEING: [
                    (GestureType.HEAD_TILT, 0.8),
                    (GestureType.LIGHT_NOD, 0.2)
                ],
                EmotionType.THINKING: [
                    (GestureType.HEAD_TILT, 0.5),
                    (GestureType.SLOW_BLINK, 0.5)
                ]
            },
            CulturalContext.LATIN_AMERICAN: {
                # Expressive, warm gestures
                EmotionType.HAPPY: [
                    (GestureType.BIG_GESTURE, 0.6),
                    (GestureType.OPEN_PALM_UP, 0.4)
                ],
                EmotionType.EXCITED: [
                    (GestureType.BOUNCY, 0.7),
                    (GestureType.EMPHATIC_HAND, 0.3)
                ],
                EmotionType.AGREEING: [
                    (GestureType.QUICK_NOD, 0.6),
                    (GestureType.OPEN_PALM, 0.4)
                ]
            },
            CulturalContext.AFRICAN: {
                # Rhythmic, community-oriented gestures
                EmotionType.HAPPY: [
                    (GestureType.BOUNCY, 0.5),
                    (GestureType.OPEN_PALM_UP, 0.5)
                ],
                EmotionType.THINKING: [
                    (GestureType.CHIN_STROKE, 0.4),
                    (GestureType.HEAD_TILT, 0.6)
                ],
                EmotionType.AGREEING: [
                    (GestureType.NOD, 0.7),
                    (GestureType.HAND_TO_CHEST, 0.3)
                ]
            }
        }
        
        # Fallback to GLOBAL for any missing cultural mappings
        for culture in CulturalContext:
            if culture != CulturalContext.GLOBAL:
                if culture not in self._gesture_mappings:
                    self._gesture_mappings[culture] = {}
                for emotion in EmotionType:
                    if emotion not in self._gesture_mappings[culture]:
                        self._gesture_mappings[culture][emotion] = self._gesture_mappings[CulturalContext.GLOBAL].get(emotion, [])
    
    def _setup_cultural_adaptations(self):
        """Initialize cultural adaptation parameters."""
        self.cultural_params = {
            CulturalContext.WESTERN: {
                'gesture_scale': 1.0,
                'personal_space': 1.2,
                'expressiveness': 0.8,
                'contact_tendency': 0.6,
                'formality_preference': 0.7,
                'eye_contact_intensity': 0.8
            },
            CulturalContext.EAST_ASIAN: {
                'gesture_scale': 0.7,
                'personal_space': 1.0,
                'expressiveness': 0.5,
                'contact_tendency': 0.3,
                'formality_preference': 0.9,
                'eye_contact_intensity': 0.4,
                'bow_tendency': 0.8
            },
            CulturalContext.MIDDLE_EASTERN: {
                'gesture_scale': 1.2,
                'personal_space': 0.8,
                'expressiveness': 1.0,
                'contact_tendency': 0.8,
                'formality_preference': 0.8,
                'eye_contact_intensity': 0.9,
                'hand_gesture_frequency': 1.3
            },
            CulturalContext.SOUTH_ASIAN: {
                'gesture_scale': 1.1,
                'personal_space': 0.9,
                'expressiveness': 0.9,
                'contact_tendency': 0.7,
                'formality_preference': 0.8,
                'eye_contact_intensity': 0.6,
                'head_movement_frequency': 1.4
            },
            CulturalContext.LATIN_AMERICAN: {
                'gesture_scale': 1.3,
                'personal_space': 0.7,
                'expressiveness': 1.1,
                'contact_tendency': 0.9,
                'formality_preference': 0.4,
                'eye_contact_intensity': 0.9,
                'warmth_factor': 1.2
            },
            CulturalContext.AFRICAN: {
                'gesture_scale': 1.1,
                'personal_space': 0.9,
                'expressiveness': 0.9,
                'contact_tendency': 0.7,
                'formality_preference': 0.6,
                'eye_contact_intensity': 0.8,
                'rhythmic_tendency': 1.2
            },
            CulturalContext.GLOBAL: {
                'gesture_scale': 1.0,
                'personal_space': 1.0,
                'expressiveness': 0.8,
                'contact_tendency': 0.5,
                'formality_preference': 0.6,
                'eye_contact_intensity': 0.7
            }
        }
    
    def set_cultural_context(self, culture: Union[CulturalContext, str]):
        """Set the cultural context for gesture generation."""
        if isinstance(culture, str):
            culture = CulturalContext[culture.upper()]
        self.params.cultural_context = culture
    
    def update_context(self, **kwargs):
        """Update the current interaction context."""
        self._context.update(kwargs)
        
    def analyze_speech_for_intensity(self, text: str, speech_metrics: Optional[Dict] = None) -> float:
        """
        Automatically determine intensity based on speech content and metrics.
        
        Args:
            text: The spoken text to analyze
            speech_metrics: Optional dict containing:
                - volume: 0.0 to 1.0
                - pitch_variation: 0.0 to 1.0
                - speech_rate: words per second
                
        Returns:
            float: Intensity value between 0.0 and 1.0
        """
        # Base intensity from speech metrics if available
        if speech_metrics:
            volume = speech_metrics.get('volume', 0.5)
            pitch_var = speech_metrics.get('pitch_variation', 0.5)
            speech_rate = speech_metrics.get('speech_rate', 3.0)  # words per second
            
            # Normalize speech rate (assuming 2-5 wps is normal range)
            rate_intensity = min(1.0, max(0.0, (speech_rate - 2) / 3.0))
            
            # Combine metrics (weighted average)
            intensity = (volume * 0.4 + pitch_var * 0.3 + rate_intensity * 0.3)
        else:
            intensity = 0.5  # Default neutral
        
        # Text-based intensity adjustments
        text = text.lower()
        
        # Punctuation analysis
        if '!' in text:
            intensity = min(1.0, intensity * 1.3)  # Excitement/emphasis
        if '?' in text:
            intensity = max(0.3, intensity * 0.9)  # Slightly more reserved for questions
            
        # Word emphasis detection (capitalized words, ALL CAPS, etc.)
        if any(word.isupper() and len(word) > 2 for word in text.split()):
            intensity = min(1.0, intensity * 1.2)
            
        # Emotional word detection (simplified example)
        emotional_words = {
            '!': 1.3, '?': 0.9,  # Punctuation
            'amazing': 1.4, 'terrible': 1.4, 'love': 1.3, 'hate': 1.5,
            'urgent': 1.3, 'important': 1.2, 'please': 1.1, 'now': 1.2,
            'maybe': 0.7, 'perhaps': 0.7, 'possibly': 0.7, 'slightly': 0.8
        }
        
        for word, modifier in emotional_words.items():
            if word in text:
                intensity = min(1.0, intensity * modifier)
                
        # Apply cultural context
        cultural_params = self.cultural_params[self.params.cultural_context]
        intensity = min(1.0, intensity * cultural_params['expressiveness'])
        
        return max(0.1, min(1.0, intensity))  # Keep within bounds
    
    def _get_contextual_intensity(self, base_intensity: float, emotion: EmotionType) -> float:
        """Adjust intensity based on context and cultural factors."""
        cultural_params = self.cultural_params[self.params.cultural_context]
        
        # Base intensity from emotion and cultural expressiveness
        intensity = base_intensity * cultural_params['expressiveness']
        
        # Adjust based on environment
        if self._context.get('environment') == 'formal':
            intensity *= 0.7
        elif self._context.get('environment') == 'casual':
            intensity *= 1.2
            
        # Adjust based on interlocutor style
        if self._context.get('interlocutor_style') == 'expressive':
            intensity = min(1.0, intensity * 1.2)
        elif self._context.get('interlocutor_style') == 'reserved':
            intensity *= 0.8
            
        return max(0.0, min(1.0, intensity))
    
    def _select_gesture_based_on_context(self, emotion: EmotionType) -> GestureType:
        """Select a gesture considering the current context and cultural norms."""
        # Get cultural mapping or fall back to global
        culture = self.params.cultural_context
        if culture not in self._gesture_mappings or emotion not in self._gesture_mappings[culture]:
            culture = CulturalContext.GLOBAL
        
        possible_gestures = self._gesture_mappings[culture][emotion]
        
        # Consider previous gestures to avoid repetition
        recent_gestures = self._context.get('previous_gestures', [])[-3:]
        
        # Filter out recently used gestures
        filtered_gestures = [
            (g, w) for g, w in possible_gestures 
            if g not in recent_gestures
        ] or possible_gestures  # Fall back to all if all are recently used
        
        # Select based on weights
        gestures, weights = zip(*filtered_gestures)
        gesture_type = random.choices(gestures, weights=weights, k=1)[0]
        
        # Update context
        if 'previous_gestures' in self._context:
            self._context['previous_gestures'].append(gesture_type)
            
        return gesture_type
    
    def get_gesture_for_emotion(self, emotion: EmotionType, intensity: float = 1.0) -> GestureDefinition:
        """
        Get an appropriate gesture for the given emotion and intensity.
        
        Args:
            emotion: The emotion to generate a gesture for
            intensity: Base intensity of the emotion (0.0 to 1.0)
            
        Returns:
            A GestureDefinition with emotion and intensity applied
        """
        # Adjust intensity based on context and culture
        adjusted_intensity = self._get_contextual_intensity(intensity, emotion)
        
        # Select gesture considering context and cultural norms
        gesture_type = self._select_gesture_based_on_context(emotion)
        
        # Create gesture definition
        gesture = GestureDefinition(gesture_type=gesture_type)
        
        # Apply emotion and adjusted intensity
        gesture.apply_emotion(emotion, adjusted_intensity * self.params.intensity)
        
        # Apply cultural scaling
        cultural_params = self.cultural_params[self.params.cultural_context]
        
        return gesture
    
    def get_gesture_sequence(self, emotion: EmotionType, duration: float, 
                           intensity: Optional[float] = None,
                           text: Optional[str] = None,
                           speech_metrics: Optional[Dict] = None,
                           context: Optional[Dict] = None) -> List[GestureDefinition]:
        """
        Generate a sequence of gestures with automatic intensity modulation.
        
        Args:
            emotion: The emotion to express
            duration: Total duration of the gesture sequence
            intensity: Optional manual intensity (0.0-1.0). If None, will be auto-calculated.
            text: Optional spoken text for intensity analysis
            speech_metrics: Optional dict with volume, pitch_variation, speech_rate
            context: Additional context parameters
            
        Returns:
            List of GestureDefinition objects
        """
        # Update context and calculate intensity if needed
        if context:
            self.update_context(**context)
            
        # Auto-calculate intensity if not provided
        if intensity is None and text:
            intensity = self.analyze_speech_for_intensity(text, speech_metrics)
        elif intensity is None:
            intensity = 0.7  # Default neutral intensity
        
        gestures = []
        remaining_duration = duration
        
        # Calculate number of gestures based on duration (1-2 gestures per second)
        target_gesture_count = max(1, min(int(duration), int(duration * 1.5)))
        avg_gesture_duration = duration / target_gesture_count
        
        # Generate gestures
        for _ in range(target_gesture_count):
            gesture = self.get_gesture_for_emotion(emotion, intensity)
            gestures.append(gesture)
            remaining_duration -= avg_gesture_duration
            
        return gestures
    
    def blend_emotions(self, emotion1: EmotionType, emotion2: EmotionType, ratio: float) -> EmotionType:
        """
avg_gesture_duration = duration / target_gesture_count
        Blend two emotions based on a ratio.
        
        Args:
            emotion1: First emotion
            emotion2: Second emotion
            ratio: Blend ratio (0.0 = all emotion1, 1.0 = all emotion2)
            
        Returns:
            The dominant emotion based on the blend ratio
        """
        # Simple blending - returns the emotion with higher weight
        return emotion2 if ratio > 0.5 else emotion1
        
        # For more sophisticated blending, we could implement a more complex system
        # that creates hybrid gestures based on both emotions
    
    def update_parameters(self, **kwargs):
        """Update gesture generation parameters."""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)

# Example usage
if __name__ == "__main__":
    # Create gesture mapper
    mapper = EmotionGestureMapper()
    
    # Example: Get gestures for different emotions
    emotions = [
        (EmotionType.HAPPY, 0.8),
        (EmotionType.ANGRY, 0.9),
        (EmotionType.SAD, 0.6),
        (EmotionType.SURPRISED, 0.7)
    ]
    
    for emotion, intensity in emotions:
        print(f"\n--- {emotion.name} (Intensity: {intensity}) ---")
        gestures = mapper.get_gesture_sequence(emotion, duration=5.0, intensity=intensity)
        
        for i, gesture in enumerate(gestures, 1):
            print(f"  Gesture {i}: {gesture.gesture_type.name}")
            print(f"    - Intensity: {gesture.intensity:.2f}")
            print(f"    - Speed: {gesture.speed:.2f}")
            print(f"    - Duration: {gesture.duration:.2f}s")
