"""
Test script for background upscaling functionality.

This script demonstrates how to use the BackgroundUpscaler class to enhance
background regions while preserving face details.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
import torch

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
    from models.background_upscaler import create_background_upscaler
except ImportError as e:
    logger.error(f"Failed to import background_upscaler: {e}")
    sys.exit(1)

def process_image(
    input_path: str,
    output_dir: str = 'output',
    upscale_factor: int = 2,
    blur_strength: float = 0.5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    """
    Process a single image with background upscaling.
    
    Args:
        input_path: Path to the input image
        output_dir: Directory to save the results
        upscale_factor: Factor to upscale the background (2 or 4)
        blur_strength: Strength of background blur (0.0 to 1.0)
        device: Device to run the model on ('cuda' or 'cpu')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to load image: {input_path}")
    
    logger.info(f"Processing image: {input_path} ({image.shape[1]}x{image.shape[0]})")
    
    try:
        # Create upscaler
        upscaler = create_background_upscaler(
            device=device,
            upscale_factor=upscale_factor,
            blur_strength=blur_strength
        )
        
        # Process image
        result = upscaler.process_frame(image)
        
        # Save results
        input_filename = Path(input_path).stem
        cv2.imwrite(
            os.path.join(output_dir, f"{input_filename}_original.jpg"),
            image
        )
        cv2.imwrite(
            os.path.join(output_dir, f"{input_filename}_upscaled_x{upscale_factor}.jpg"),
            result
        )
        
        # Create side-by-side comparison
        h, w = image.shape[:2]
        comparison = np.hstack((
            cv2.resize(image, (w // 2, h // 2)),
            cv2.resize(result, (w // 2, h // 2))
        ))
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            comparison, f'Upscaled x{upscale_factor}', 
            (w // 2 + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        comparison_path = os.path.join(
            output_dir, 
            f"{input_filename}_comparison_x{upscale_factor}.jpg"
        )
        cv2.imwrite(comparison_path, comparison)
        logger.info(f"Saved results to {os.path.abspath(comparison_path)}")
        
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise

def process_video(
    input_path: str,
    output_dir: str = 'output',
    upscale_factor: int = 2,
    blur_strength: float = 0.5,
    max_frames: int = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    """
    Process a video with background upscaling.
    
    Args:
        input_path: Path to the input video
        output_dir: Directory to save the results
        upscale_factor: Factor to upscale the background (2 or 4)
        blur_strength: Strength of background blur (0.0 to 1.0)
        max_frames: Maximum number of frames to process (for testing)
        device: Device to run the model on ('cuda' or 'cpu')
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            frame_count = min(frame_count, max_frames)
        
        logger.info(
            f"Processing video: {input_path} ({width}x{height}, {fps:.2f} fps, {frame_count} frames)"
        )
        
        # Create upscaler
        upscaler = create_background_upscaler(
            device=device,
            upscale_factor=upscale_factor,
            blur_strength=blur_strength
        )
        
        # Create output video
        output_path = os.path.join(
            output_dir,
            f"{Path(input_path).stem}_upscaled_x{upscale_factor}.mp4"
        )
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )
        
        try:
            # Process frames
            for i in tqdm(range(frame_count), desc="Processing frames"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = upscaler.process_frame(frame)
                
                # Write frame to output video
                out.write(processed_frame)
                
                # Save sample frames for comparison
                if i == 0 or i == frame_count // 2 or i == frame_count - 1:
                    # Create side-by-side comparison
                    h, w = frame.shape[:2]
                    comparison = np.hstack((
                        cv2.resize(frame, (w // 2, h // 2)),
                        cv2.resize(processed_frame, (w // 2, h // 2))
                    ))
                    
                    # Add labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA
                    )
                    cv2.putText(
                        comparison, f'Upscaled x{upscale_factor}', 
                        (w // 2 + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA
                    )
                    
                    sample_path = os.path.join(
                        output_dir, 
                        f"{Path(input_path).stem}_frame_{i:04d}_comparison.jpg"
                    )
                    cv2.imwrite(sample_path, comparison)
                
                # Clear CUDA cache to prevent memory issues
                if 'cuda' in device:
                    torch.cuda.empty_cache()
        
        finally:
            # Release resources
            cap.release()
            out.release()
        
        logger.info(f"Saved enhanced video to {os.path.abspath(output_path)}")
    
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test background upscaling')
    parser.add_argument('input', type=str, help='Path to input image or video')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--upscale', type=int, default=2, choices=[2, 4], help='Upscale factor (2 or 4)')
    parser.add_argument('--blur', type=float, default=0.5, help='Background blur strength (0.0 to 1.0)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to run the model on (cuda or cpu)')
    parser.add_argument('--max-frames', type=int, default=None, 
                       help='Maximum number of frames to process (video only)')
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Check if input is image or video
    input_ext = Path(args.input).suffix.lower()
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']
    
    try:
        if input_ext in image_exts:
            process_image(
                input_path=args.input,
                output_dir=args.output_dir,
                upscale_factor=args.upscale,
                blur_strength=args.blur,
                device=args.device
            )
        elif input_ext in video_exts:
            process_video(
                input_path=args.input,
                output_dir=args.output_dir,
                upscale_factor=args.upscale,
                blur_strength=args.blur,
                max_frames=args.max_frames,
                device=args.device
            )
        else:
            logger.error(f"Unsupported file format: {input_ext}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
