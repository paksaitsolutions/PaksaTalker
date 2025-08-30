"""
Real Text-to-Speech Service Implementation
Supports multiple TTS providers and voice models
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TTSService:
    """Real TTS service with multiple provider support"""
    
    def __init__(self):
        self.providers = {}
        self._init_providers()
    
    def _init_providers(self):
        """Initialize available TTS providers"""
        
        # Azure Cognitive Services
        try:
            import azure.cognitiveservices.speech as speechsdk
            self.providers['azure'] = self._init_azure()
        except ImportError:
            logger.warning("Azure Speech SDK not available")
        
        # Google Cloud TTS
        try:
            from google.cloud import texttospeech
            self.providers['google'] = self._init_google()
        except ImportError:
            logger.warning("Google Cloud TTS not available")
        
        # Amazon Polly
        try:
            import boto3
            self.providers['aws'] = self._init_aws()
        except ImportError:
            logger.warning("AWS Polly not available")
        
        # ElevenLabs (Premium)
        try:
            import elevenlabs
            self.providers['elevenlabs'] = self._init_elevenlabs()
        except ImportError:
            logger.warning("ElevenLabs not available")
    
    def _init_azure(self):
        """Initialize Azure Speech Services"""
        import azure.cognitiveservices.speech as speechsdk
        
        speech_key = os.getenv('AZURE_SPEECH_KEY')
        service_region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
        
        if not speech_key:
            logger.warning("Azure Speech key not configured")
            return None
        
        speech_config = speechsdk.SpeechConfig(
            subscription=speech_key, 
            region=service_region
        )
        return speech_config
    
    def _init_google(self):
        """Initialize Google Cloud TTS"""
        from google.cloud import texttospeech
        
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not credentials_path:
            logger.warning("Google Cloud credentials not configured")
            return None
        
        return texttospeech.TextToSpeechClient()
    
    def _init_aws(self):
        """Initialize AWS Polly"""
        import boto3
        
        try:
            return boto3.client('polly')
        except Exception as e:
            logger.warning(f"AWS Polly initialization failed: {e}")
            return None
    
    def _init_elevenlabs(self):
        """Initialize ElevenLabs"""
        import elevenlabs
        
        api_key = os.getenv('ELEVENLABS_API_KEY')
        if not api_key:
            logger.warning("ElevenLabs API key not configured")
            return None
        
        elevenlabs.set_api_key(api_key)
        return True
    
    def generate_speech(
        self, 
        text: str, 
        voice: str = "en-US-JennyNeural",
        output_path: Optional[str] = None,
        provider: str = "auto"
    ) -> str:
        """Generate speech from text using specified provider"""
        
        if provider == "auto":
            provider = self._select_best_provider(voice)
        
        if provider == "azure" and "azure" in self.providers:
            return self._generate_azure(text, voice, output_path)
        elif provider == "google" and "google" in self.providers:
            return self._generate_google(text, voice, output_path)
        elif provider == "aws" and "aws" in self.providers:
            return self._generate_aws(text, voice, output_path)
        elif provider == "elevenlabs" and "elevenlabs" in self.providers:
            return self._generate_elevenlabs(text, voice, output_path)
        else:
            # Fallback to any available provider
            for available_provider in self.providers:
                try:
                    return getattr(self, f"_generate_{available_provider}")(text, voice, output_path)
                except Exception as e:
                    logger.warning(f"Provider {available_provider} failed: {e}")
                    continue
            
            raise RuntimeError("No TTS providers available")
    
    def _select_best_provider(self, voice: str) -> str:
        """Select best provider based on voice model"""
        
        # Azure voices
        if voice.endswith("Neural"):
            return "azure"
        
        # ElevenLabs voices
        if voice.startswith("eleven_"):
            return "elevenlabs"
        
        # Google voices
        if "Wavenet" in voice or "Standard" in voice:
            return "google"
        
        # AWS voices
        if voice in ["Joanna", "Matthew", "Amy", "Brian"]:
            return "aws"
        
        # Default to Azure
        return "azure"
    
    def _generate_azure(self, text: str, voice: str, output_path: Optional[str]) -> str:
        """Generate speech using Azure"""
        import azure.cognitiveservices.speech as speechsdk
        
        if not output_path:
            output_path = tempfile.mktemp(suffix='.wav')
        
        speech_config = self.providers['azure']
        speech_config.speech_synthesis_voice_name = voice
        
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, 
            audio_config=audio_config
        )
        
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return output_path
        else:
            raise RuntimeError(f"Azure TTS failed: {result.reason}")
    
    def _generate_google(self, text: str, voice: str, output_path: Optional[str]) -> str:
        """Generate speech using Google Cloud TTS"""
        from google.cloud import texttospeech
        
        if not output_path:
            output_path = tempfile.mktemp(suffix='.wav')
        
        client = self.providers['google']
        
        # Parse voice name (format: language-voice-gender)
        parts = voice.split('-')
        language_code = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else "en-US"
        voice_name = voice if len(parts) >= 3 else "en-US-Wavenet-D"
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )
        
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config
        )
        
        with open(output_path, 'wb') as out:
            out.write(response.audio_content)
        
        return output_path
    
    def _generate_aws(self, text: str, voice: str, output_path: Optional[str]) -> str:
        """Generate speech using AWS Polly"""
        if not output_path:
            output_path = tempfile.mktemp(suffix='.wav')
        
        polly = self.providers['aws']
        
        # Map voice names
        voice_mapping = {
            "en-US-JennyNeural": "Joanna",
            "en-US-ChristopherNeural": "Matthew",
            "en-GB-SoniaNeural": "Amy",
            "en-GB-RyanNeural": "Brian"
        }
        
        polly_voice = voice_mapping.get(voice, "Joanna")
        
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat='pcm',
            VoiceId=polly_voice,
            Engine='neural' if 'Neural' in voice else 'standard'
        )
        
        with open(output_path, 'wb') as out:
            out.write(response['AudioStream'].read())
        
        return output_path
    
    def _generate_elevenlabs(self, text: str, voice: str, output_path: Optional[str]) -> str:
        """Generate speech using ElevenLabs"""
        import elevenlabs
        
        if not output_path:
            output_path = tempfile.mktemp(suffix='.wav')
        
        # Map voice names
        voice_mapping = {
            "en-US-JennyNeural": "Rachel",
            "en-US-ChristopherNeural": "Josh",
            "en-GB-SoniaNeural": "Bella",
            "en-GB-RyanNeural": "Antoni"
        }
        
        elevenlabs_voice = voice_mapping.get(voice, "Rachel")
        
        audio = elevenlabs.generate(
            text=text,
            voice=elevenlabs_voice,
            model="eleven_monolingual_v1"
        )
        
        with open(output_path, 'wb') as out:
            out.write(audio)
        
        return output_path
    
    def get_available_voices(self, provider: str = "all") -> Dict[str, Any]:
        """Get list of available voices"""
        voices = {}
        
        if provider in ["all", "azure"] and "azure" in self.providers:
            voices["azure"] = self._get_azure_voices()
        
        if provider in ["all", "google"] and "google" in self.providers:
            voices["google"] = self._get_google_voices()
        
        if provider in ["all", "aws"] and "aws" in self.providers:
            voices["aws"] = self._get_aws_voices()
        
        if provider in ["all", "elevenlabs"] and "elevenlabs" in self.providers:
            voices["elevenlabs"] = self._get_elevenlabs_voices()
        
        return voices
    
    def _get_azure_voices(self) -> list:
        """Get Azure voices"""
        return [
            {"id": "en-US-JennyNeural", "name": "Jenny", "language": "en-US", "gender": "Female"},
            {"id": "en-US-ChristopherNeural", "name": "Christopher", "language": "en-US", "gender": "Male"},
            {"id": "en-GB-SoniaNeural", "name": "Sonia", "language": "en-GB", "gender": "Female"},
            {"id": "en-GB-RyanNeural", "name": "Ryan", "language": "en-GB", "gender": "Male"}
        ]
    
    def _get_google_voices(self) -> list:
        """Get Google voices"""
        return [
            {"id": "en-US-Wavenet-D", "name": "Wavenet D", "language": "en-US", "gender": "Male"},
            {"id": "en-US-Wavenet-F", "name": "Wavenet F", "language": "en-US", "gender": "Female"}
        ]
    
    def _get_aws_voices(self) -> list:
        """Get AWS voices"""
        return [
            {"id": "Joanna", "name": "Joanna", "language": "en-US", "gender": "Female"},
            {"id": "Matthew", "name": "Matthew", "language": "en-US", "gender": "Male"}
        ]
    
    def _get_elevenlabs_voices(self) -> list:
        """Get ElevenLabs voices"""
        return [
            {"id": "Rachel", "name": "Rachel", "language": "en-US", "gender": "Female"},
            {"id": "Josh", "name": "Josh", "language": "en-US", "gender": "Male"}
        ]

# Global TTS service instance
_tts_service = None

def get_tts_service() -> TTSService:
    """Get or create global TTS service instance"""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service