import os
import sys
import yaml
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class GestureIntegrator:
    def __init__(self, config_path: str = None):
        """
        Initialize the Gesture Integrator with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.config.get('force_cpu', False) else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize models
        self._init_sadtalker()
        self._init_stylegestures()
        
        # Create output directories
        os.makedirs(self.config['paths']['output_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['temp_dir'], exist_ok=True)
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _init_sadtalker(self):
        """Initialize SadTalker model components."""
        print("Initializing SadTalker...")
        try:
            from src.facerender.animate import AnimateFromCoeff
            from src.generate_batch import get_data
            from src.utils.preprocess import CropAndExtract
            
            # Initialize SadTalker components
            self.animate_from_coeff = AnimateFromCoeff(
                device=self.device,
                sadtalker_path=self.config['paths']['sadtalker_root']
            )
            
            self.crop_extractor = CropAndExtract(
                self.config['sadtalker']['checkpoint'],
                self.device
            )
            
            self.get_data = get_data
            print("SadTalker initialized successfully.")
            
        except Exception as e:
            print(f"Error initializing SadTalker: {str(e)}")
            raise
    
    def _init_stylegestures(self):
        """Initialize StyleGestures model."""
        print("Initializing StyleGestures...")
        try:
            # Add StyleGestures to path
            stylegestures_path = Path(self.config['paths']['stylegestures_root'])
            if stylegestures_path.exists():
                sys.path.append(str(stylegestures_path))
            
            # Import StyleGestures
            from stylegestures import StyleGestures
            
            # Initialize model with CPU/GPU handling
            model_config = self.config['stylegestures'].copy()
            model_config['device'] = str(self.device)  # Convert device to string for compatibility
            self.style_gestures = StyleGestures(**model_config)
            print("StyleGestures initialized successfully.")
            
        except ImportError as e:
            print(f"Warning: Could not import StyleGestures. Gesture generation will be disabled. {str(e)}")
            self.style_gestures = None
        except Exception as e:
            print(f"Error initializing StyleGestures: {str(e)}")
            self.style_gestures = None
    
    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """Process audio file and extract features."""
        print(f"Processing audio: {audio_path}")
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        target_sample_rate = 16000  # Common sample rate for speech processing
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate
            )
            waveform = resampler(waveform)
        
        # Extract audio features (MFCC, pitch, energy, etc.)
        # This is a simplified example - you might need to adjust based on your needs
        mfcc = torchaudio.compliance.kaldi.mfcc(
            waveform,
            sample_frequency=target_sample_rate,
            use_energy=True
        )
        
        return {
            'waveform': waveform,
            'sample_rate': target_sample_rate,
            'mfcc': mfcc,
            'duration': waveform.shape[1] / target_sample_rate
        }
    
    def generate_gestures(self, audio_features: Dict[str, Any], style: Optional[Dict] = None) -> torch.Tensor:
        """Generate gestures from audio features."""
        if self.style_gestures is None:
            print("Warning: StyleGestures not available. Returning dummy gestures.")
            # Return dummy gestures (this is just a placeholder)
            num_frames = int(audio_features['duration'] * 30)  # 30 FPS
            return torch.zeros((num_frames, 3))  # x, y, z coordinates
        
        # Use default style if not provided
        if style is None:
            style = {
                'style': self.config['stylegestures']['style'],
                'intensity': self.config['stylegestures']['intensity']
            }
        
        # Generate gestures using StyleGestures
        gestures = self.style_gestures.generate(
            audio=audio_features['waveform'],
            style=style
        )
        
        return gestures
    
    def generate_video(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        style: Optional[Dict] = None
    ) -> str:
        """
        Generate a talking head video with synchronized gestures.
        
        Args:
            image_path: Path to the source image
            audio_path: Path to the input audio
            output_path: Path to save the output video
            style: Optional style parameters for gesture generation
            
        Returns:
            Path to the generated video
        """
        print(f"Starting video generation for {image_path} with audio {audio_path}")
        
        # Process audio
        audio_features = self.process_audio(audio_path)
        
        # Generate gestures
        gestures = self.generate_gestures(audio_features, style)
        
        # Generate facial animation using SadTalker
        print("Generating facial animation...")
        # This is a simplified example - you'll need to adapt it to your specific SadTalker implementation
        try:
            # Get face crops and landmarks
            first_frame_dir = os.path.join(self.config['paths']['temp_dir'], 'first_frame_dir')
            os.makedirs(first_frame_dir, exist_ok=True)
            
            # Preprocess the image
            first_coeff_path, crop_pic_path, crop_info = self.crop_extractor.generate(
                image_path, 
                first_frame_dir, 
                self.config['sadtalker']['preprocess'],
                True,
                self.config['sadtalker']['size']
            )
            
            # Generate animation coefficients
            # Note: You'll need to implement this part based on your SadTalker version
            # This is a placeholder for the actual implementation
            coeff_output_path = os.path.join(self.config['paths']['temp_dir'], 'coeffs.npy')
            
            # Animate using the coefficients
            self.animate_from_coeff.generate(
                first_coeff_path,
                audio_path,
                crop_pic_path,
                crop_info,
                output_path,
                self.config['sadtalker']['pose_style'],
                self.config['sadtalker']['expression_scale'],
                self.config['sadtalker']['input_yaw'],
                self.config['sadtalker']['input_pitch'],
                self.config['sadtalker']['input_roll'],
                self.config['sadtalker']['enhancer'],
                self.config['sadtalker']['background_enhancer'],
                self.config['sadtalker']['up_scale']
            )
            
            print(f"Video generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating video: {str(e)}")
            raise

def main():
    """Main function for testing the GestureIntegrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate talking head video with gestures')
    parser.add_argument('--image', type=str, required=True, help='Path to source image')
    parser.add_argument('--audio', type=str, required=True, help='Path to input audio')
    parser.add_argument('--output', type=str, default='output/result.mp4', help='Output video path')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    
    args = parser.parse_args()
    
    try:
        # Initialize the integrator
        integrator = GestureIntegrator(args.config)
        
        # Generate the video
        output_path = integrator.generate_video(
            image_path=args.image,
            audio_path=args.audio,
            output_path=args.output
        )
        
        print(f"Video generation completed. Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
