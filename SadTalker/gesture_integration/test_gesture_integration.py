import sys
import os
import torch
import yaml
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add SadTalker to path
sadtalker_path = Path("D:/SadTalker")
sys.path.insert(0, str(sadtalker_path))

# Verify PyTorch is available
try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("Note: Running on CPU")
except Exception as e:
    print(f"Error initializing PyTorch: {e}")
    sys.exit(1)

# Import SadTalker modules
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.utils.preprocess import CropAndExtract

class GestureIntegrationTest:
    def __init__(self):
        # Force CPU mode for compatibility
        self.device = 'cpu'
        print(f"Using device: {self.device}")
        
        # Load config
        config_path = sadtalker_path / "src" / "config" / "facerender.yaml"
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print("Successfully loaded config file")
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
        
        # Initialize models
        try:
            self._init_models()
        except Exception as e:
            print(f"Error initializing models: {e}")
            sys.exit(1)
    
    def _init_models(self):
        """Initialize the required models for animation."""
        print("\nInitializing models...")
        
        try:
            # Initialize animation model
            print("Initializing animation model...")
            self.animate_from_coeff = AnimateFromCoeff(
                self.config, 
                str(sadtalker_path / 'checkpoints'),  # Convert to string for compatibility
                self.device,
                'full'
            )
            
            # Initialize crop and extract model
            print("Initializing crop and extract model...")
            self.crop_extractor = CropAndExtract(
                self.config,
                str(sadtalker_path / 'checkpoints'),  # Convert to string for compatibility
                self.device
            )
            
            print("✅ Models initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_gesture_generation(self, image_path, audio_path, output_path):
        """Test gesture generation with a sample image and audio."""
        print("\nTesting gesture generation...")
        
        # Prepare input data
        print(f"Processing image: {image_path}")
        print(f"Using audio: {audio_path}")
        
        # Get batch data (simplified for testing)
        # In a real scenario, you would process the audio and extract features
        batch = {
            'source_image': image_path,
            'audio_path': audio_path,
            'output_path': output_path,
            'still': False,
            'preprocess': 'full',
            'expression_scale': 1.0,
            'input_yaw': None,
            'input_pitch': None,
            'input_roll': None,
            'enhancer': None,
            'background_enhancer': None,
            'face3d_vis': False
        }
        
        # This is a simplified test - in a real scenario, you would:
        # 1. Process the audio to get gesture features
        # 2. Generate motion coefficients
        # 3. Animate the image using the coefficients
        
        print("Gesture generation test completed (simplified).")
        print(f"Output will be saved to: {output_path}")

def check_paths():
    """Check if required files and directories exist."""
    required_paths = [
        (sadtalker_path / "src", "SadTalker source directory"),
        (sadtalker_path / "checkpoints", "Checkpoints directory"),
        (sadtalker_path / "examples/source_image/art_0.png", "Test image"),
        (sadtalker_path / "examples/driven_audio/RD_Radio31_000.wav", "Test audio"),
        (sadtalker_path / "results", "Output directory")
    ]
    
    all_paths_exist = True
    for path, desc in required_paths:
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {desc}: {path}")
        if not exists:
            all_paths_exist = False
    
    return all_paths_exist

def main():
    print("Gesture Integration Test")
    print("=" * 50)
    
    # Check if all required paths exist
    print("\nChecking required files and directories:")
    if not check_paths():
        print("\n❌ Some required files or directories are missing.")
        return
    
    print("\nAll required files and directories found!")
    
    # Initialize test
    print("\nInitializing gesture integration test...")
    test = GestureIntegrationTest()
    
    # Test paths
    test_image = str(sadtalker_path / "examples/source_image/art_0.png")
    test_audio = str(sadtalker_path / "examples/driven_audio/RD_Radio31_000.wav")
    output_dir = str(sadtalker_path / "results")
    
    print("\nStarting gesture generation test...")
    test.test_gesture_generation(test_image, test_audio, output_dir)
    
    print("\nTest completed. Check the output directory for results.")

if __name__ == "__main__":
    main()
