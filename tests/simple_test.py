"""Simple test script for background upscaler."""
import cv2
import numpy as np
import torch
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a simple test image
def create_test_image():
    """Create a simple test image with a face and background."""
    # Create a 256x256 black image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Add a white face (circle)
    cv2.circle(img, (128, 128), 50, (255, 255, 255), -1)
    
    # Add some text
    cv2.putText(img, "TEST", (100, 130), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return img

def main():
    """Run a simple test of the background upscaler."""
    try:
        # Create test directories
        test_dir = Path("test_images")
        test_dir.mkdir(exist_ok=True)
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Create and save test image
        test_img = create_test_image()
        test_img_path = test_dir / "test_face.jpg"
        cv2.imwrite(str(test_img_path), test_img)
        logger.info(f"Created test image at: {test_img_path}")
        
        # Try to import the background upscaler
        try:
            from models.background_upscaler import BackgroundUpscaler
            logger.info("Successfully imported BackgroundUpscaler")
        except Exception as e:
            logger.error(f"Failed to import BackgroundUpscaler: {e}")
            return
        
        # Initialize the upscaler
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            
            upscaler = BackgroundUpscaler(device=device)
            logger.info("Initialized BackgroundUpscaler")
            
            # Process the test image
            logger.info("Processing test image...")
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            
            # Process the image
            processed = upscaler.process(img_rgb)
            
            # Convert back to BGR for saving
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            
            # Save the result
            output_path = output_dir / "processed_test.jpg"
            cv2.imwrite(str(output_path), processed_bgr)
            logger.info(f"Saved processed image to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    main()
