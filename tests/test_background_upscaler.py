"""Test script for BackgroundUpscaler class."""
import cv2
import numpy as np
import torch
from pathlib import Path
import logging
import traceback
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('background_upscaler_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow importing from the models directory
sys.path.append(str(Path(__file__).parent))

# Check OpenCV version and CUDA support
logger.info(f"OpenCV version: {cv2.__version__}")
logger.info(f"CUDA available: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")

try:
    from models.background_upscaler import create_background_upscaler
    logger.info("Successfully imported background_upscaler")
except ImportError as e:
    logger.error(f"Failed to import background_upscaler: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

def main():
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Output directory: {output_dir.absolute()}")
        
        # Input image path
        input_path = Path("test_images/person.jpg")
        logger.debug(f"Input image path: {input_path.absolute()}")
        
        # Check if input image exists
        if not input_path.exists():
            logger.error(f"Input image not found: {input_path.absolute()}")
            # Create a test image if none exists
            logger.info("Creating a test image...")
            test_image = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.putText(test_image, "Test Image", (50, 128), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            input_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(input_path), test_image)
            logger.info(f"Created test image at: {input_path.absolute()}")
        
        # Initialize the background upscaler
        logger.info("Initializing background upscaler...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        logger.info(f"Torch CUDA available: {torch.cuda.is_available()}")
        
        try:
            upscaler = create_background_upscaler(
                device=device,
                upscale_factor=2,
                blur_strength=0.5
            )
            logger.info("Background upscaler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize background upscaler: {e}")
            logger.error(traceback.format_exc())
            return
        
        # Read the input image
        logger.info(f"Reading input image: {input_path}")
        try:
            image = cv2.imread(str(input_path))
            if image is None:
                raise ValueError("cv2.imread returned None")
            logger.info(f"Image loaded successfully. Shape: {image.shape}")
        except Exception as e:
            logger.error(f"Failed to read image {input_path}: {e}")
            logger.error(traceback.format_exc())
            return
        
        # Process the image
        logger.info("Processing image...")
        try:
            # Convert BGR to RGB (OpenCV uses BGR by default)
            logger.debug("Converting BGR to RGB...")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            logger.debug("Processing image with background upscaler...")
            processed = upscaler.process(image_rgb)
            logger.info("Image processing completed")
            
            # Convert back to BGR for saving
            logger.debug("Converting RGB back to BGR...")
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            
            # Save the result
            output_path = output_dir / "background_upscaled.jpg"
            logger.debug(f"Saving result to: {output_path}")
            cv2.imwrite(str(output_path), processed_bgr)
            logger.info(f"Successfully saved result to: {output_path.absolute()}")
            
            # Also save the original for comparison
            cv2.imwrite(str(output_dir / "original.jpg"), image)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.critical(f"Unexpected error in main: {e}")
        logger.critical(traceback.format_exc())

if __name__ == "__main__":
    import torch  # Import torch here to avoid circular imports
    main()
