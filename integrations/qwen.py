'''Qwen integration for PaksaTalker'''
import os
import torch
from typing import Optional, Dict, Any, List, Union
import json
from pathlib import Path

from integrations.base import BaseIntegration
from config import config

# Import Qwen with error handling
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

class QwenIntegration(BaseIntegration):
    '''Integration with Qwen language model for enhanced text processing'''

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Omni-7B", device: str = None):
        '''Initialize Qwen integration

        Args:
            model_name: Name or path of the Qwen model to load
            device: Device to run the model on
        '''
        super().__init__(device)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.initialized = False

    def load_model(self, **kwargs) -> None:
        '''Load the Qwen model and tokenizer'''
        if not QWEN_AVAILABLE:
            raise ImportError("Qwen dependencies not available. Please install transformers and torch.")

        if not self.initialized:
            try:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    **kwargs.get('tokenizer_kwargs', {})
                )

                # Load model with appropriate device map
                device_map = "auto"
                if self.device == "cuda" and torch.cuda.is_available():
                    device_map = "cuda"
                elif self.device == "mps" and torch.backends.mps.is_available():
                    device_map = "mps"

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=device_map,
                    trust_remote_code=True,
                    **{
                        'torch_dtype': torch.float16 if 'cuda' in device_map else torch.float32,
                        **kwargs.get('model_kwargs', {})
                    }
                )

                # Set generation config
                self.model.generation_config = GenerationConfig.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )

                self.initialized = True

            except Exception as e:
                raise RuntimeError(f"Failed to load Qwen model: {e}")

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        '''Generate text from a prompt

        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated text
            temperature: Sampling temperature (lower = more focused, higher = more random)
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        '''
        if not self.initialized:
            self.load_model()

        try:
            # Encode the prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  # Adjust based on model's max length
            )

            # Move inputs to the same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate response
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

            # Decode and return the generated text
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            raise RuntimeError(f"Text generation failed: {e}")

    def enhance_prompt(
        self,
        original_prompt: str,
        style: str = "professional",
        **kwargs
    ) -> str:
        '''Enhance a prompt with more detail and clarity

        Args:
            original_prompt: The original user prompt
            style: Desired style for enhancement (e.g., 'professional', 'casual', 'detailed')
            **kwargs: Additional parameters for generation

        Returns:
            Enhanced prompt text
        '''
        enhancement_instruction = {
            "professional": "Enhance the following prompt to be more professional and detailed: ",
            "casual": "Make this prompt more conversational and friendly: ",
            "detailed": "Add more specific details and context to this prompt: "
        }.get(style.lower(), "Improve this prompt: ")

        enhanced_prompt = self.generate(
            f"{enhancement_instruction}{original_prompt}",
            max_length=300,
            temperature=0.7,
            **kwargs
        )

        return enhanced_prompt

    def analyze_video_requirements(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        '''Analyze a text prompt to determine video generation requirements

        Args:
            prompt: User's prompt describing the desired video'
            **kwargs: Additional parameters for analysis

        Returns:
            Dictionary containing analysis results with keys like 'style', 'mood', 'gestures', etc.
        '''
        analysis_prompt = f"""Analyze the following video generation request and extract key parameters.
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
                analysis_prompt,
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
        '''Check if Qwen is loaded'''
        return self.initialized and self.model is not None

    def unload(self) -> None:
        '''Unload Qwen and free resources'''
        if hasattr(self, 'model') and self.model is not None:
            # Free up GPU memory
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        super().unload()

    def __del__(self):
        '''Cleanup on object deletion'''
        self.unload()
