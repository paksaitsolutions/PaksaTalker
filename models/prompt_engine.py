"""
Advanced Prompt Engineering System for PaksaTalker
Real implementation with system prompts, few-shot learning, and safety filters
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PersonaType(Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    CUSTOMER_SERVICE = "customer_service"
    HEALTHCARE = "healthcare"
    TECHNICAL = "technical"

class SafetyLevel(Enum):
    STRICT = "strict"
    MODERATE = "moderate"
    RELAXED = "relaxed"

@dataclass
class PromptTemplate:
    """Template for few-shot learning examples"""
    input_text: str
    expected_output: str
    context: str = ""
    emotion: str = "neutral"
    style: str = "professional"

@dataclass
class SystemPrompt:
    """System prompt configuration for consistent persona"""
    persona: PersonaType
    role_description: str
    behavioral_guidelines: List[str]
    response_format: str
    constraints: List[str]
    examples: List[PromptTemplate] = field(default_factory=list)

class AdvancedPromptEngine:
    """Advanced prompt engineering with real AI integration"""
    
    def __init__(self):
        self.system_prompts = self._initialize_system_prompts()
        self.few_shot_templates = self._initialize_few_shot_templates()
        self.safety_filters = self._initialize_safety_filters()
        self.moderation_patterns = self._load_moderation_patterns()
        
    def _initialize_system_prompts(self) -> Dict[PersonaType, SystemPrompt]:
        """Initialize system prompts for different personas"""
        return {
            PersonaType.PROFESSIONAL: SystemPrompt(
                persona=PersonaType.PROFESSIONAL,
                role_description="You are a professional AI presenter delivering clear, authoritative content.",
                behavioral_guidelines=[
                    "Maintain formal tone and professional language",
                    "Use structured, logical presentation",
                    "Include relevant data and facts when appropriate",
                    "Avoid casual expressions and slang",
                    "Speak with confidence and authority"
                ],
                response_format="Structured with clear introduction, main points, and conclusion",
                constraints=[
                    "Keep responses between 30-180 seconds when spoken",
                    "Use professional vocabulary",
                    "Maintain objective perspective",
                    "Include actionable insights"
                ]
            ),
            
            PersonaType.EDUCATIONAL: SystemPrompt(
                persona=PersonaType.EDUCATIONAL,
                role_description="You are an engaging educator who makes complex topics accessible.",
                behavioral_guidelines=[
                    "Break down complex concepts into simple terms",
                    "Use analogies and examples to illustrate points",
                    "Encourage learning and curiosity",
                    "Ask rhetorical questions to engage audience",
                    "Build knowledge progressively"
                ],
                response_format="Educational with clear explanations and examples",
                constraints=[
                    "Use age-appropriate language",
                    "Include learning objectives",
                    "Provide practical applications",
                    "Encourage further exploration"
                ]
            ),
            
            PersonaType.CASUAL: SystemPrompt(
                persona=PersonaType.CASUAL,
                role_description="You are a friendly, approachable communicator having a natural conversation.",
                behavioral_guidelines=[
                    "Use conversational tone and natural language",
                    "Include personal touches and relatability",
                    "Show enthusiasm and genuine interest",
                    "Use appropriate humor when suitable",
                    "Be warm and engaging"
                ],
                response_format="Conversational with natural flow and personal connection",
                constraints=[
                    "Maintain authenticity",
                    "Keep it relatable and accessible",
                    "Use everyday language",
                    "Show personality while staying professional"
                ]
            ),
            
            PersonaType.CUSTOMER_SERVICE: SystemPrompt(
                persona=PersonaType.CUSTOMER_SERVICE,
                role_description="You are a helpful customer service representative focused on solving problems.",
                behavioral_guidelines=[
                    "Show empathy and understanding",
                    "Provide clear, actionable solutions",
                    "Maintain patience and professionalism",
                    "Acknowledge concerns and validate feelings",
                    "Follow up with additional assistance offers"
                ],
                response_format="Solution-focused with empathetic communication",
                constraints=[
                    "Always offer help and alternatives",
                    "Use positive language",
                    "Provide step-by-step guidance",
                    "Ensure customer satisfaction"
                ]
            ),
            
            PersonaType.HEALTHCARE: SystemPrompt(
                persona=PersonaType.HEALTHCARE,
                role_description="You are a healthcare communicator providing clear, accurate health information.",
                behavioral_guidelines=[
                    "Use clear, medical terminology when appropriate",
                    "Emphasize evidence-based information",
                    "Show compassion and understanding",
                    "Encourage professional medical consultation",
                    "Maintain confidentiality and privacy"
                ],
                response_format="Informative with medical accuracy and compassionate delivery",
                constraints=[
                    "Never provide specific medical diagnoses",
                    "Always recommend consulting healthcare professionals",
                    "Use scientifically accurate information",
                    "Show sensitivity to health concerns"
                ]
            )
        }
    
    def _initialize_few_shot_templates(self) -> Dict[str, List[PromptTemplate]]:
        """Initialize few-shot learning templates for different scenarios"""
        return {
            "product_presentation": [
                PromptTemplate(
                    input_text="Introduce our new smartphone with advanced camera features",
                    expected_output="Today I'm excited to present our revolutionary new smartphone that's about to change how you capture life's moments. With our breakthrough 108-megapixel camera system and AI-powered night mode, you'll create stunning photos in any lighting condition. Let me show you what makes this device truly special.",
                    context="product_launch",
                    emotion="excited",
                    style="professional"
                ),
                PromptTemplate(
                    input_text="Explain the benefits of our cloud storage service",
                    expected_output="Our cloud storage solution offers you the freedom to access your files anywhere, anytime. With military-grade encryption and 99.9% uptime guarantee, your data stays secure and always available. Whether you're collaborating with teams or backing up precious memories, we've got you covered.",
                    context="service_explanation",
                    emotion="confident",
                    style="professional"
                )
            ],
            
            "educational_content": [
                PromptTemplate(
                    input_text="Explain photosynthesis to middle school students",
                    expected_output="Imagine plants as tiny solar-powered factories! Just like how solar panels convert sunlight into electricity, plants use their green leaves to capture sunlight and turn it into food. They take in carbon dioxide from the air we breathe out, combine it with water from their roots, and create sugar - their food! As a bonus, they release oxygen back into the air for us to breathe. It's nature's perfect recycling system!",
                    context="science_education",
                    emotion="enthusiastic",
                    style="educational"
                ),
                PromptTemplate(
                    input_text="Teach basic programming concepts",
                    expected_output="Think of programming like giving directions to a very literal friend. You need to be extremely specific about every step. A variable is like a labeled box where you store information. A function is like a recipe - you give it ingredients (inputs) and it follows steps to create something new (output). And loops? They're like saying 'repeat this 10 times' - the computer will do exactly that!",
                    context="programming_basics",
                    emotion="encouraging",
                    style="educational"
                )
            ],
            
            "customer_service": [
                PromptTemplate(
                    input_text="Address a billing complaint",
                    expected_output="I completely understand your frustration with the billing issue, and I sincerely apologize for any inconvenience this has caused. Let me personally look into this right away and make sure we resolve it quickly. I'm going to review your account details and ensure you receive the correct billing going forward. Your satisfaction is our top priority.",
                    context="complaint_resolution",
                    emotion="empathetic",
                    style="customer_service"
                ),
                PromptTemplate(
                    input_text="Explain a product return policy",
                    expected_output="I'd be happy to help you with our return policy! You have 30 days from purchase to return any item in its original condition. Simply bring your receipt and the item to any of our stores, or use our prepaid return label for online purchases. If you're not completely satisfied, we'll provide a full refund or exchange. Is there a specific item you'd like to return today?",
                    context="policy_explanation",
                    emotion="helpful",
                    style="customer_service"
                )
            ]
        }
    
    def _initialize_safety_filters(self) -> Dict[SafetyLevel, Dict[str, Any]]:
        """Initialize safety and moderation filters"""
        return {
            SafetyLevel.STRICT: {
                "blocked_topics": [
                    "violence", "hate_speech", "harassment", "self_harm", 
                    "illegal_activities", "adult_content", "misinformation",
                    "personal_attacks", "discrimination", "extremism"
                ],
                "content_guidelines": [
                    "No controversial political statements",
                    "No medical advice or diagnoses",
                    "No financial investment advice",
                    "No personal information requests",
                    "No inappropriate language"
                ],
                "response_modifications": {
                    "add_disclaimers": True,
                    "soften_language": True,
                    "add_professional_tone": True
                }
            },
            
            SafetyLevel.MODERATE: {
                "blocked_topics": [
                    "violence", "hate_speech", "harassment", "self_harm",
                    "illegal_activities", "adult_content"
                ],
                "content_guidelines": [
                    "Avoid controversial topics without context",
                    "No specific medical or financial advice",
                    "Maintain respectful discourse"
                ],
                "response_modifications": {
                    "add_disclaimers": False,
                    "soften_language": False,
                    "add_professional_tone": False
                }
            },
            
            SafetyLevel.RELAXED: {
                "blocked_topics": [
                    "hate_speech", "harassment", "illegal_activities"
                ],
                "content_guidelines": [
                    "Maintain basic respect and civility"
                ],
                "response_modifications": {
                    "add_disclaimers": False,
                    "soften_language": False,
                    "add_professional_tone": False
                }
            }
        }
    
    def _load_moderation_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for content moderation"""
        return {
            "profanity": [
                r'\b(damn|hell|crap)\b',  # Mild profanity
                r'\b(shit|fuck|bitch)\b',  # Strong profanity
            ],
            "personal_info": [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                r'\b\d{3}-\d{3}-\d{4}\b',  # Phone pattern
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            ],
            "hate_speech": [
                r'\b(hate|despise|loathe)\s+(all|every)\s+\w+',
                r'\b(kill|destroy|eliminate)\s+(all|every)\s+\w+',
            ],
            "medical_advice": [
                r'\b(diagnose|prescribe|treat|cure)\b',
                r'\byou\s+(have|need|should\s+take)\s+\w+\s+(medication|drug|pill)\b',
            ]
        }
    
    def construct_dynamic_prompt(
        self,
        user_input: str,
        persona: PersonaType = PersonaType.PROFESSIONAL,
        context: str = "",
        emotion: str = "neutral",
        safety_level: SafetyLevel = SafetyLevel.MODERATE,
        include_examples: bool = True,
        max_length: int = 150
    ) -> str:
        """Dynamically construct optimized prompt with all features"""
        
        # 1. Safety check first
        if not self._passes_safety_filter(user_input, safety_level):
            return self._generate_safety_response(user_input, safety_level)
        
        # 2. Get system prompt for persona
        system_prompt = self.system_prompts[persona]
        
        # 3. Select relevant few-shot examples
        examples = self._select_relevant_examples(user_input, context, persona)
        
        # 4. Construct the full prompt
        prompt_parts = []
        
        # System instruction
        prompt_parts.append(f"SYSTEM: {system_prompt.role_description}")
        prompt_parts.append(f"BEHAVIORAL GUIDELINES: {'; '.join(system_prompt.behavioral_guidelines)}")
        prompt_parts.append(f"RESPONSE FORMAT: {system_prompt.response_format}")
        prompt_parts.append(f"CONSTRAINTS: {'; '.join(system_prompt.constraints)}")
        
        # Context and emotion
        if context:
            prompt_parts.append(f"CONTEXT: {context}")
        prompt_parts.append(f"EMOTION: {emotion}")
        prompt_parts.append(f"TARGET LENGTH: {max_length} words maximum")
        
        # Few-shot examples
        if include_examples and examples:
            prompt_parts.append("EXAMPLES:")
            for i, example in enumerate(examples[:2], 1):  # Limit to 2 examples
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append(f"Input: {example.input_text}")
                prompt_parts.append(f"Output: {example.expected_output}")
        
        # User input
        prompt_parts.append(f"USER INPUT: {user_input}")
        prompt_parts.append("RESPONSE:")
        
        return "\n".join(prompt_parts)
    
    def _passes_safety_filter(self, text: str, safety_level: SafetyLevel) -> bool:
        """Check if text passes safety filters"""
        safety_config = self.safety_filters[safety_level]
        
        # Check blocked topics
        text_lower = text.lower()
        for topic in safety_config["blocked_topics"]:
            if topic.replace("_", " ") in text_lower:
                return False
        
        # Check moderation patterns
        for category, patterns in self.moderation_patterns.items():
            if category in safety_config["blocked_topics"]:
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        return False
        
        return True
    
    def _generate_safety_response(self, user_input: str, safety_level: SafetyLevel) -> str:
        """Generate appropriate safety response"""
        safety_responses = {
            SafetyLevel.STRICT: "I'm not able to create content on that topic. Let me help you with something else that would be more appropriate for a professional presentation.",
            SafetyLevel.MODERATE: "I'd prefer to focus on more constructive topics. How about we explore a different subject for your video?",
            SafetyLevel.RELAXED: "Let's try a different approach to that topic that would work better for video content."
        }
        return safety_responses[safety_level]
    
    def _select_relevant_examples(
        self, 
        user_input: str, 
        context: str, 
        persona: PersonaType
    ) -> List[PromptTemplate]:
        """Select most relevant few-shot examples"""
        all_examples = []
        
        # Collect examples from all categories
        for category, examples in self.few_shot_templates.items():
            all_examples.extend(examples)
        
        # Score examples by relevance
        scored_examples = []
        user_words = set(user_input.lower().split())
        
        for example in all_examples:
            score = 0
            example_words = set(example.input_text.lower().split())
            
            # Word overlap score
            overlap = len(user_words.intersection(example_words))
            score += overlap * 2
            
            # Context match
            if context and context in example.context:
                score += 5
            
            # Persona compatibility (simplified)
            persona_compatibility = {
                PersonaType.PROFESSIONAL: ["product_presentation", "service_explanation"],
                PersonaType.EDUCATIONAL: ["science_education", "programming_basics"],
                PersonaType.CUSTOMER_SERVICE: ["complaint_resolution", "policy_explanation"]
            }
            
            if persona in persona_compatibility:
                if any(ctx in example.context for ctx in persona_compatibility[persona]):
                    score += 3
            
            scored_examples.append((score, example))
        
        # Return top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example for score, example in scored_examples[:3]]
    
    def apply_safety_moderation(self, generated_text: str, safety_level: SafetyLevel) -> str:
        """Apply post-generation safety moderation"""
        safety_config = self.safety_filters[safety_level]
        moderated_text = generated_text
        
        # Apply response modifications
        if safety_config["response_modifications"]["add_disclaimers"]:
            moderated_text = self._add_disclaimers(moderated_text)
        
        if safety_config["response_modifications"]["soften_language"]:
            moderated_text = self._soften_language(moderated_text)
        
        if safety_config["response_modifications"]["add_professional_tone"]:
            moderated_text = self._add_professional_tone(moderated_text)
        
        return moderated_text
    
    def _add_disclaimers(self, text: str) -> str:
        """Add appropriate disclaimers to content"""
        disclaimers = {
            "medical": "Please consult with a healthcare professional for medical advice.",
            "financial": "This is for informational purposes only and not financial advice.",
            "legal": "This information is general in nature and not legal advice."
        }
        
        text_lower = text.lower()
        for topic, disclaimer in disclaimers.items():
            if any(word in text_lower for word in [topic, f"{topic} advice", f"{topic} information"]):
                text += f" {disclaimer}"
        
        return text
    
    def _soften_language(self, text: str) -> str:
        """Soften potentially harsh language"""
        replacements = {
            r'\bmust\b': 'should consider',
            r'\bwrong\b': 'not ideal',
            r'\bbad\b': 'less effective',
            r'\bfailed?\b': 'didn\'t work as expected',
            r'\bimpossible\b': 'very challenging'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _add_professional_tone(self, text: str) -> str:
        """Add professional tone markers"""
        if not text.strip():
            return text
        
        # Add professional opening if needed
        if not any(text.startswith(phrase) for phrase in ["Thank you", "I'd like to", "Let me", "Today"]):
            text = f"I'd like to share some insights about {text[0].lower()}{text[1:]}"
        
        return text
    
    def generate_persona_prompt(
        self,
        topic: str,
        persona: PersonaType,
        duration_seconds: int = 60,
        emotion: str = "neutral",
        context: str = ""
    ) -> str:
        """Generate a complete prompt for specific persona and requirements"""
        
        # Calculate target word count (average 150 words per minute)
        target_words = int((duration_seconds / 60) * 150)
        
        prompt = self.construct_dynamic_prompt(
            user_input=f"Create a {duration_seconds}-second presentation about {topic}",
            persona=persona,
            context=context,
            emotion=emotion,
            safety_level=SafetyLevel.MODERATE,
            include_examples=True,
            max_length=target_words
        )
        
        return prompt
    
    def validate_and_enhance_output(
        self,
        generated_text: str,
        persona: PersonaType,
        safety_level: SafetyLevel = SafetyLevel.MODERATE
    ) -> Dict[str, Any]:
        """Validate and enhance generated output"""
        
        # Apply safety moderation
        moderated_text = self.apply_safety_moderation(generated_text, safety_level)
        
        # Quality checks
        quality_score = self._calculate_quality_score(moderated_text, persona)
        
        # Enhancement suggestions
        enhancements = self._suggest_enhancements(moderated_text, persona)
        
        return {
            "original_text": generated_text,
            "moderated_text": moderated_text,
            "quality_score": quality_score,
            "enhancements": enhancements,
            "safety_passed": self._passes_safety_filter(generated_text, safety_level),
            "word_count": len(moderated_text.split()),
            "estimated_duration": len(moderated_text.split()) / 2.5  # ~150 WPM
        }
    
    def _calculate_quality_score(self, text: str, persona: PersonaType) -> float:
        """Calculate quality score based on persona requirements"""
        score = 0.0
        system_prompt = self.system_prompts[persona]
        
        # Length appropriateness (0-25 points)
        word_count = len(text.split())
        if 50 <= word_count <= 200:
            score += 25
        elif 30 <= word_count <= 250:
            score += 20
        else:
            score += 10
        
        # Persona adherence (0-25 points)
        persona_keywords = {
            PersonaType.PROFESSIONAL: ["professional", "data", "results", "analysis", "strategy"],
            PersonaType.EDUCATIONAL: ["learn", "understand", "example", "concept", "explain"],
            PersonaType.CASUAL: ["you", "we", "like", "really", "great"],
            PersonaType.CUSTOMER_SERVICE: ["help", "assist", "support", "solution", "service"]
        }
        
        if persona in persona_keywords:
            keyword_matches = sum(1 for keyword in persona_keywords[persona] if keyword in text.lower())
            score += min(25, keyword_matches * 5)
        
        # Structure and clarity (0-25 points)
        sentences = text.split('.')
        if 3 <= len(sentences) <= 8:
            score += 25
        elif 2 <= len(sentences) <= 10:
            score += 20
        else:
            score += 10
        
        # Engagement factors (0-25 points)
        engagement_indicators = ["you", "your", "we", "our", "?", "!", "imagine", "consider"]
        engagement_count = sum(1 for indicator in engagement_indicators if indicator in text.lower())
        score += min(25, engagement_count * 3)
        
        return min(100.0, score)
    
    def _suggest_enhancements(self, text: str, persona: PersonaType) -> List[str]:
        """Suggest enhancements based on persona and content analysis"""
        suggestions = []
        
        word_count = len(text.split())
        if word_count < 30:
            suggestions.append("Consider adding more detail and examples")
        elif word_count > 200:
            suggestions.append("Consider condensing for better impact")
        
        if persona == PersonaType.EDUCATIONAL and "?" not in text:
            suggestions.append("Add rhetorical questions to increase engagement")
        
        if persona == PersonaType.PROFESSIONAL and not any(word in text.lower() for word in ["data", "results", "analysis"]):
            suggestions.append("Include more concrete facts or data points")
        
        if "!" not in text and "?" not in text:
            suggestions.append("Add more dynamic punctuation for emphasis")
        
        return suggestions

    # New: style adaptation utilities
    def emotion_embedding(self, emotion: str) -> List[float]:
        """Return a simple emotion embedding vector for downstream use."""
        base = {
            'neutral': [0.0, 0.0, 0.0, 0.0],
            'happy':   [1.0, 0.0, 0.2, 0.1],
            'sad':     [0.0, 1.0, 0.1, 0.0],
            'angry':   [0.0, 0.0, 1.0, 0.2],
            'excited': [0.8, 0.0, 0.2, 0.8],
            'serious': [0.0, 0.6, 0.1, 0.0]
        }
        return base.get(emotion.lower(), base['neutral'])

    def adapt_text_style(
        self,
        text: str,
        formality: str = 'neutral',  # casual | neutral | formal
        domain: Optional[str] = None,
        personality: Optional[str] = None,
        emotion: Optional[str] = None
    ) -> str:
        """Adapt text for formality, domain terminology, personality traits, and emotion cues."""
        adapted = text.strip()

        # Apply formality
        if formality == 'formal':
            replacements = {
                "can't": "cannot", "won't": "will not", "it's": "it is",
                "gonna": "going to", "wanna": "want to"
            }
            for k,v in replacements.items():
                adapted = adapted.replace(k, v).replace(k.title(), v)
        elif formality == 'casual':
            adapted = adapted.replace("do not", "don't").replace("cannot", "can't")

        # Inject domain terminology (simple heuristic list)
        domain_terms = {
            'medical': ["clinical evidence", "protocol", "diagnostic"],
            'finance': ["portfolio", "liquidity", "risk management"],
            'tech':    ["scalability", "latency", "architecture"],
            'education': ["scaffolding", "learning outcomes", "assessment"]
        }
        if domain and domain.lower() in domain_terms:
            terms = domain_terms[domain.lower()]
            if terms and terms[0] not in adapted.lower():
                adapted += f"\n\nNote: Key terms — {', '.join(terms)}."

        # Personality traits
        if personality:
            traits_map = {
                'friendly': "Maintain a warm, approachable tone.",
                'authoritative': "Use confident, decisive language.",
                'enthusiastic': "Add positive energy and encouragement.",
                'concise': "Be succinct and to the point."
            }
            tip = traits_map.get(personality.lower())
            if tip:
                adapted += f"\n\nStyle: {tip}"

        # Emotion cues
        if emotion:
            cues = {
                'happy': "[SMILE]",
                'serious': "[SERIOUS]",
                'excited': "[EXCITED]",
                'neutral': "[NEUTRAL]"
            }
            cue = cues.get(emotion.lower())
            if cue and cue not in adapted:
                adapted = f"{cue} {adapted}"

        return adapted

# Global instance
prompt_engine = AdvancedPromptEngine()
