"""
Qwen2.5-Omni-7B Real Implementation
Multimodal AI model for text, audio, and image processing
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline
)

# Try to import AutoProcessor, fallback if not available
try:
    from transformers import AutoProcessor
    HAS_PROCESSOR = True
except ImportError:
    HAS_PROCESSOR = False
    AutoProcessor = None
import numpy as np
import librosa
from PIL import Image
import io
import base64
from typing import Optional, Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)

class QwenOmniModel:
    """Real Qwen2.5-Omni-7B implementation with multimodal capabilities"""
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-Omni-7B", device: str = "auto"):
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.audio_pipeline = None
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the Qwen2.5-Omni model and components"""
        try:
            logger.info(f"Loading Qwen2.5-Omni-7B on {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Load processor for multimodal inputs
            if HAS_PROCESSOR and AutoProcessor:
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_path,
                        trust_remote_code=True
                    )
                except Exception as e:
                    logger.warning(f"Could not load processor: {e}")
            else:
                logger.warning("AutoProcessor not available, multimodal features limited")
            
            # Initialize audio pipeline
            try:
                self.audio_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-base",
                    device=0 if self.device == "cuda" else -1
                )
            except Exception as e:
                logger.warning(f"Could not load audio pipeline: {e}")
            
            logger.info("Qwen2.5-Omni model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-Omni model: {e}")
            raise
    
    def generate_text(
        self, 
        prompt: str, 
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text response from prompt with safety filtering"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Apply safety filtering if available
            try:
                from models.prompt_engine import prompt_engine
                if not prompt_engine._passes_safety_filter(prompt, prompt_engine.SafetyLevel.MODERATE):
                    return prompt_engine._generate_safety_response(prompt, prompt_engine.SafetyLevel.MODERATE)
            except ImportError:
                pass
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def process_audio(self, audio_data: Union[str, np.ndarray, bytes]) -> str:
        """Process audio input and convert to text"""
        if not self.audio_pipeline:
            return "Audio processing not available"
        
        try:
            # Handle different audio input types
            if isinstance(audio_data, str):
                # File path
                audio_array, sr = librosa.load(audio_data, sr=16000)
            elif isinstance(audio_data, bytes):
                # Audio bytes
                audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=16000)
            else:
                # NumPy array
                audio_array = audio_data
            
            # Transcribe audio
            result = self.audio_pipeline(audio_array)
            return result.get('text', '')
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return f"Error processing audio: {str(e)}"
    
    def process_image(self, image_data: Union[str, Image.Image, bytes]) -> str:
        """Process image input and generate description"""
        try:
            # Handle different image input types
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    # Base64 encoded image
                    image_data = base64.b64decode(image_data.split(',')[1])
                    image = Image.open(io.BytesIO(image_data))
                else:
                    # File path
                    image = Image.open(image_data)
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = image_data
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate image description using the model
            prompt = "Describe this image in detail:"
            
            if self.processor:
                # Use multimodal processor if available
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
            else:
                # Fallback to text-only description
                response = "Image processing requires multimodal processor"
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return f"Error processing image: {str(e)}"
    
    def multimodal_chat(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Image.Image, bytes]] = None,
        audio: Optional[Union[str, np.ndarray, bytes]] = None,
        context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Process multimodal input and generate response"""
        
        # Build conversation context
        conversation = []
        if context:
            conversation.extend(context)
        
        # Process audio input
        audio_text = ""
        if audio:
            audio_text = self.process_audio(audio)
            if audio_text:
                conversation.append({
                    "role": "user",
                    "content": f"[Audio transcription]: {audio_text}"
                })
        
        # Process image input
        image_desc = ""
        if image:
            image_desc = self.process_image(image)
            if image_desc:
                conversation.append({
                    "role": "user", 
                    "content": f"[Image description]: {image_desc}"
                })
        
        # Add text input
        if text:
            conversation.append({
                "role": "user",
                "content": text
            })
        
        # Build prompt from conversation
        prompt_parts = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role.title()}: {content}")
        
        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        
        # Generate response
        response = self.generate_text(prompt, max_length=512)
        
        return {
            "response": response,
            "audio_transcription": audio_text,
            "image_description": image_desc,
            "conversation": conversation
        }
    
    def generate_avatar_script(
        self,
        topic: str,
        style: str = "professional",
        duration: int = 30,
        audience: str = "general"
    ) -> Dict[str, Any]:
        """Generate script specifically for avatar presentation"""
        
        prompt = f"""
Create a {duration}-second avatar presentation script about: {topic}

Style: {style}
Target audience: {audience}

Requirements:
- Natural, conversational tone
- Clear pronunciation-friendly text
- Appropriate pacing for {duration} seconds
- Engaging and informative
- Include natural pauses [PAUSE]
- Add emotion cues [SMILE], [SERIOUS], [EXCITED]

Script:
"""
        
        script = self.generate_text(prompt, max_length=400)
        
        # Estimate timing (average 150 words per minute)
        word_count = len(script.split())
        estimated_duration = (word_count / 150) * 60
        
        return {
            "script": script,
            "word_count": word_count,
            "estimated_duration": estimated_duration,
            "style": style,
            "topic": topic
        }

# Global instance
_qwen_model = None

def get_qwen_model() -> QwenOmniModel:
    """Get or create global Qwen model instance"""
    global _qwen_model
    if _qwen_model is None:
        _qwen_model = QwenOmniModel()
        _qwen_model.load_model()
    return _qwen_model

def generate_text_response(prompt: str, **kwargs) -> str:
    """Quick text generation function"""
    model = get_qwen_model()
    return model.generate_text(prompt, **kwargs)

def process_multimodal_input(
    text: Optional[str] = None,
    image: Optional[Union[str, Image.Image, bytes]] = None,
    audio: Optional[Union[str, np.ndarray, bytes]] = None
) -> Dict[str, Any]:
    """Quick multimodal processing function"""
    model = get_qwen_model()
    return model.multimodal_chat(text=text, image=image, audio=audio)