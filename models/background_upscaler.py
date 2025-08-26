"""
Background Upscaling Module

This module provides functionality for upscaling and enhancing background regions
in videos while preserving face regions. It uses a combination of traditional
image processing and deep learning techniques for optimal results.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import logging

from .base import BaseModel
from .super_resolution import FaceSuperResolution

logger = logging.getLogger(__name__)

class BackgroundUpscaler(BaseModel):
    """
    Background upscaler for enhancing non-face regions in videos.
    Uses a combination of traditional and deep learning-based approaches.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the BackgroundUpscaler.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        super().__init__()
        self.device = device
        self.upscale_factor = 2  # Default upscale factor
        self.face_detector = self._init_face_detector()
        self.super_res = self._init_super_resolution()
        self.blur_strength = 0.5  # 0.0 to 1.0
        
    def _init_face_detector(self):
        """Initialize face detection model with fallback to Haar Cascade."""
        try:
            # First try: OpenCV's DNN face detector
            try:
                import urllib.request
                weights_dir = Path(__file__).parent / 'weights'
                weights_dir.mkdir(exist_ok=True, parents=True)
                
                # Use local paths for the model files
                proto_path = str(weights_dir / 'deploy.prototxt')
                model_path = str(weights_dir / 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
                
                # Download model files if they don't exist
                if not Path(proto_path).exists():
                    logger.info("Downloading face detection model...")
                    proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
                    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
                    
                    try:
                        urllib.request.urlretrieve(proto_url, proto_path)
                        urllib.request.urlretrieve(model_url, model_path)
                    except Exception as e:
                        logger.warning(f"Failed to download face detection model: {e}")
                        raise
                
                if not Path(proto_path).exists() or not Path(model_path).exists():
                    raise FileNotFoundError("Face detection model files not found")
                
                net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                if net is None:
                    raise RuntimeError("Failed to load face detection model")
                    
                if 'cuda' in self.device and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    logger.info("Using DNN-based face detector with CUDA")
                else:
                    logger.info("Using DNN-based face detector (CPU)")
                    
                return net
                
            except Exception as e:
                logger.warning(f"DNN face detector failed: {e}. Trying Haar Cascade...")
                
            # Fallback: Haar Cascade
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if not Path(cascade_path).exists():
                    # Try to find the cascade file in other possible locations
                    possible_paths = [
                        Path(cv2.__file__).parent / 'data/haarcascade_frontalface_default.xml',
                        Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml',
                        Path('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'),
                        Path('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
                    )]
                    
                    for path in possible_paths:
                        if path.exists():
                            cascade_path = str(path)
                            break
                    else:
                        raise FileNotFoundError("Haar Cascade file not found in any standard location")
                
                cascade = cv2.CascadeClassifier(cascade_path)
                if cascade.empty():
                    raise RuntimeError("Failed to load Haar Cascade classifier")
                    
                logger.info("Using Haar Cascade face detector")
                return cascade
                
            except Exception as e:
                logger.warning(f"Haar Cascade face detector failed: {e}")
                return None
            
        except Exception as e:
            logger.warning(f"All face detection methods failed: {e}")
            logger.warning("Face detection will be disabled. Background upscaling will affect entire image.")
            return None
    
    def _init_super_resolution(self):
        """Initialize super-resolution model."""
        try:
            # Use our existing FaceSuperResolution model
            return FaceSuperResolution(device=self.device)
        except Exception as e:
            logger.error(f"Failed to initialize super-resolution: {e}")
            return None
    
    def create_face_mask(self, image: np.ndarray, expand: float = 0.2) -> np.ndarray:
        """
        Create a mask for face regions in the image.
        
        Args:
            image: Input image in BGR format
            expand: Percentage to expand the face bounding boxes
            
        Returns:
            Binary mask where 255 indicates face regions
        """
        if self.face_detector is None:
            return np.zeros_like(image[:, :, 0])
            
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Detect faces
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Expand the bounding box
                width = endX - startX
                height = endY - startY
                startX = max(0, int(startX - width * expand))
                startY = max(0, int(startY - height * expand))
                endX = min(w, int(endX + width * expand))
                endY = min(h, int(endY + height * expand))
                
                # Add to mask
                cv2.rectangle(mask, (startX, startY), (endX, endY), 255, -1)
        
        # Apply Gaussian blur for smoother edges
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        return mask
    
    def upscale_background(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Upscale the background region of an image.
        
        Args:
            image: Input image in BGR format
            mask: Optional mask where 255 indicates regions to preserve (e.g., faces)
            
        Returns:
            Image with upscaled background
        """
        if mask is None:
            mask = self.create_face_mask(image)
        
        # Convert mask to float32 in range [0, 1]
        mask_float = mask.astype(np.float32) / 255.0
        
        # Create inverse mask for background
        bg_mask = 1.0 - mask_float
        
        # Upscale the entire image
        if self.super_res:
            upscaled = self.super_res.process_frame(image)
        else:
            # Fallback to bicubic interpolation
            upscaled = cv2.resize(
                image, 
                None, 
                fx=self.upscale_factor, 
                fy=self.upscale_factor, 
                interpolation=cv2.INTER_CUBIC
            )
            # Resize back to original dimensions for blending
            upscaled = cv2.resize(upscaled, (image.shape[1], image.shape[0]))
        
        # Apply blur to the upscaled background
        blurred_bg = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=3, sigmaY=3)
        
        # Blend the original and upscaled images using the mask
        result = image.copy()
        for c in range(3):
            result[:, :, c] = (
                image[:, :, c] * mask_float + 
                blurred_bg[:, :, c] * bg_mask * (1 - self.blur_strength) +
                upscaled[:, :, c] * bg_mask * self.blur_strength
            )
        
        return result.astype(np.uint8)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame to enhance the background.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Frame with enhanced background
        """
        # Create face mask
        mask = self.create_face_mask(frame)
        
        # Process background
        return self.upscale_background(frame, mask)
    
    def enhance_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_resolution: Tuple[int, int] = (1920, 1080),
        fps: Optional[float] = None,
        batch_size: int = 1
    ) -> str:
        """
        Enhance the background of a video.
        
        Args:
            input_path: Path to the input video file
            output_path: Path to save the enhanced video
            target_resolution: Target resolution as (width, height)
            fps: Frames per second for the output video
            batch_size: Number of frames to process in parallel
            
        Returns:
            Path to the enhanced video file
        """
        from ..utils.video_utils import extract_frames, combine_frames_to_video
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process frames
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames = extract_frames(
            str(input_path),
            str(frames_dir),
            target_fps=fps,
            max_frames=None,
            overwrite=False
        )
        
        if not frames:
            raise ValueError("No frames extracted from the input video")
        
        # Process frames
        enhanced_frames = []
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"Failed to read frame: {frame_path}")
                continue
                
            # Process the frame
            enhanced_frame = self.process_frame(frame)
            
            # Resize to target resolution if needed
            if (frame.shape[1], frame.shape[0]) != target_resolution:
                enhanced_frame = cv2.resize(
                    enhanced_frame,
                    target_resolution,
                    interpolation=cv2.INTER_LANCZOS4
                )
            
            enhanced_frames.append(enhanced_frame)
        
        # Combine frames into video
        if not enhanced_frames:
            raise RuntimeError("No frames were processed successfully")
        
        # Use the first frame to get video properties
        height, width = enhanced_frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps or 30.0,  # Default to 30 FPS if not specified
            (width, height)
        )
        
        # Write frames to video
        for frame in enhanced_frames:
            out.write(frame)
        
        # Release resources
        out.release()
        
        return str(output_path)


def create_background_upscaler(
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    upscale_factor: int = 2,
    blur_strength: float = 0.5
) -> BackgroundUpscaler:
    """
    Create a background upscaler with the specified parameters.
    
    Args:
        device: Device to run the model on ('cuda' or 'cpu')
        upscale_factor: Upscaling factor (2 or 4)
        blur_strength: Strength of background blur (0.0 to 1.0)
        
    Returns:
        Configured BackgroundUpscaler instance
    """
    upscaler = BackgroundUpscaler(device=device)
    upscaler.upscale_factor = upscale_factor
    upscaler.blur_strength = max(0.0, min(1.0, blur_strength))  # Clamp to [0, 1]
    return upscaler
