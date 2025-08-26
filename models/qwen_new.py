"""Qwen language model implementation for PaksaTalker."""
import os
import torch
import json
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .base import BaseModel
from config import config

class QwenModel(BaseModel):
    """Qwen language model for text generation."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the Qwen model.
        
        Args:
            device: Device to run the model on ('cuda', 'mps', 'cpu').
        """
        super().__init__(device)
        self.model = None
        self.tokenizer = None
        self.initialized = False
    
    def load_model(self, model_name: Optional[str] = None, **kwargs) -> None:
        """Load the Qwen model.
        
        Args:
            model_name: Name or path of the model to load.
            **kwargs: Additional arguments for model loading.
        """
        if self.initialized:
            return
            
        try:
            model_name = model_name or config.get('models.qwen.model_name', 'Qwen/Qwen-7B-Chat')
            
            # Set up device
            device_map = "auto"
            if self.device == "cuda" and torch.cuda.is_available():
                device_map = "cuda"
            elif self.device == "mps" and torch.backends.mps.is_available():
                device_map = "mps"
                
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                **kwargs.get('tokenizer_kwargs', {})
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16 if 'cuda' in device_map else torch.float32,
                **kwargs.get('model_kwargs', {})
            )
            
            self.model.generation_config = GenerationConfig.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.initialized = True
            
        except Exception as e:
            self.initialized = False
            raise RuntimeError(f"Failed to load Qwen model: {e}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt text.
            max_length: Maximum length of generated text.
            temperature: Sampling temperature (lower = more focused, higher = more random).
            top_p: Nucleus sampling parameter.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated text.
        """
        if not self.initialized:
            self.load_model()
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {e}")
    
    def analyze_video_requirements(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze a text prompt to determine video generation requirements.
        
        Args:
            prompt: User's prompt describing the desired video.
            **kwargs: Additional parameters for analysis.
            
        Returns:
            Dictionary containing analysis results with keys like 'style', 'mood', 'gestures', etc.
        """
        analysis_prompt = """Analyze the following video generation request and extract key parameters.
        Return a JSON object with these fields:
        - style: The overall style of the video (e.g., 'professional', 'casual', 'dramatic')
        - mood: The emotional tone (e.g., 'happy', 'serious', 'excited')
        - gestures: Type of gestures to include (e.g., 'subtle', 'expressive', 'none')
        - background: Suggested background type (e.g., 'blurred', 'office', 'outdoor')
        - duration: Estimated duration in seconds
        - additional_notes: Any other relevant notes
        
        Request: {prompt}
        
        Respond with JSON only, no additional text."""
        
        try:
            response = self.generate(
                analysis_prompt.format(prompt=prompt),
                max_length=512,
                temperature=0.3,
                **kwargs
            )
            
            # Clean up the response to ensure it's valid JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[response.find('{'):response.rfind('}')+1]
            elif response.startswith('```'):
                response = response[response.find('{'):response.rfind('}')+1]
            
            return json.loads(response)
            
        except Exception as e:
            # Return defaults if analysis fails
            return {
                'style': 'professional',
                'mood': 'neutral',
                'gestures': 'subtle',
                'background': 'blurred',
                'duration': 10,
                'additional_notes': 'Default parameters used due to analysis failure'
            }
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.initialized and self.model is not None and self.tokenizer is not None
    
    def unload(self) -> None:
        """Unload the model and free up memory."""
        if hasattr(self, 'model') and self.model is not None:
            # Free up GPU memory
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.tokenizer = None
        self.initialized = False
