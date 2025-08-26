"""Test script for ArtifactReduction functionality."""
import cv2
import numpy as np
import torch
from pathlib import Path
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow importing from the models directory
import sys
sys.path.append(str(Path(__file__).parent))

try:
    from models.artifact_reduction import create_artifact_reducer
except ImportError as e:
    logger.error(f"Failed to import artifact_reduction: {e}")
    sys.exit(1)

def process_image(
    input_path: str,
    output_dir: str = 'output',
    use_deep: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    """
    Process a single image to reduce artifacts.
    
    Args:
        input_path: Path to the input image
        output_dir: Directory to save the results
        use_deep: Whether to use deep learning model if available
        device: Device to run the model on ('cuda' or 'cpu')
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Read the input image
    logger.info(f"Reading input image: {input_path}")
    image = cv2.imread(input_path)
    if image is None:
        logger.error(f"Failed to read image: {input_path}")
        return
    
    # Initialize the artifact reducer
    logger.info("Initializing artifact reducer...")
    try:
        reducer = create_artifact_reducer(
            device=device,
            use_deep=use_deep
        )
        logger.info(f"Using device: {device}")
        logger.info(f"Using {'deep learning' if use_deep else 'traditional'} method")
        
        # Process the image
        logger.info("Processing image...")
        processed = reducer.process_frame(
            frame=image,
            use_deep=use_deep
        )
        
        # Save the result
        output_path = output_dir / f"reduced_{Path(input_path).name}"
        cv2.imwrite(str(output_path), processed)
        logger.info(f"Saved result to: {output_path}")
        
        # Also save the original for comparison
        cv2.imwrite(str(output_dir / f"original_{Path(input_path).name}"), image)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test artifact reduction')
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--no-deep', action='store_true', help='Disable deep learning model')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Process the image
    process_image(
        input_path=args.input,
        output_dir=args.output_dir,
        use_deep=not args.no_deep,
        device=args.device
    )
