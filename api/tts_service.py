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
            azure_cfg = self._init_azure()
            if azure_cfg is not None:
                self.providers['azure'] = azure_cfg
        except ImportError:
            logger.warning("Azure Speech SDK not available")
        
        # gTTS (Free) - register early to avoid cloud deps
        try:
            from gtts import gTTS  # noqa: F401
            self.providers['gtts'] = True
            logger.info("gTTS provider enabled (free)")
        except Exception:
            logger.warning("gTTS not available; install with 'pip install gTTS'")

        # Google Cloud TTS (guard all exceptions to avoid protobuf runtime issues)
        try:
            from google.cloud import texttospeech  # type: ignore
            google_cli = self._init_google()
            if google_cli is not None:
                self.providers['google'] = google_cli
        except Exception as e:
            logger.warning(f"Google Cloud TTS not available ({e})")
        
        # Amazon Polly
        try:
            import boto3
            aws_cli = self._init_aws()
            if aws_cli is not None:
                self.providers['aws'] = aws_cli
        except ImportError:
            logger.warning("AWS Polly not available")
        
        # ElevenLabs (Premium)
        try:
            import elevenlabs  # type: ignore
            el_ok = self._init_elevenlabs()
            if el_ok:
                self.providers['elevenlabs'] = True
        except Exception:
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
            # Prefer free gTTS when available, otherwise select by voice
            if 'gtts' in self.providers:
                provider = 'gtts'
            else:
                provider = self._select_best_provider(voice)
        
        if provider == "azure" and "azure" in self.providers:
            return self._generate_azure(text, voice, output_path)
        elif provider == "google" and "google" in self.providers:
            return self._generate_google(text, voice, output_path)
        elif provider == "aws" and "aws" in self.providers:
            return self._generate_aws(text, voice, output_path)
        elif provider == "elevenlabs" and "elevenlabs" in self.providers:
            return self._generate_elevenlabs(text, voice, output_path)
        elif provider == "gtts" and "gtts" in self.providers:
            return self._generate_gtts(text, voice, output_path)
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

    def _generate_gtts(self, text: str, voice: str, output_path: Optional[str]) -> str:
        """Generate speech using gTTS (free). Writes MP3, converts to WAV if requested."""
        from gtts import gTTS
        import subprocess, shutil, tempfile

        # Derive language from voice (e.g., en-US-... -> en or en-us)
        lang = "en"
        try:
            if "-" in voice:
                parts = voice.split("-")
                # take first part as language code
                lang = (parts[0] or "en").lower()
        except Exception:
            pass

        # Create temp mp3
        mp3_path = tempfile.mktemp(suffix='.mp3')
        tts = gTTS(text=text, lang=lang)
        tts.save(mp3_path)

        # If output_path not specified, return MP3 directly
        if not output_path:
            return mp3_path

        # If requested a WAV, try to convert using ffmpeg; otherwise copy MP3 to path
        if str(output_path).lower().endswith('.wav'):
            ffmpeg = shutil.which('ffmpeg')
            if ffmpeg:
                wav_path = output_path
                cmd = [ffmpeg, '-y', '-i', mp3_path, '-ar', '22050', '-ac', '1', wav_path]
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode == 0:
                        return wav_path
                    else:
                        logger.warning(f"ffmpeg conversion to wav failed: {res.stderr}")
                except Exception as e:
                    logger.warning(f"ffmpeg execution failed: {e}")
            # Fallback: just return the mp3 if conversion fails
            return mp3_path
        else:
            # Ensure directory exists then copy MP3
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(mp3_path, 'rb') as src, open(output_path, 'wb') as dst:
                    dst.write(src.read())
                return output_path
            except Exception:
                return mp3_path
    
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
