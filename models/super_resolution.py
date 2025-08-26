"""
Super Resolution Module for enhancing video quality

This module provides functionality for upscaling and enhancing video quality,
with a focus on face regions. It includes support for various super-resolution
models and post-processing techniques.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import logging

from .base import BaseModel
from ..utils.video_utils import extract_frames, combine_frames_to_video

logger = logging.getLogger(__name__)

class FaceSuperResolution(BaseModel):
    """
    Face Super-Resolution model for enhancing face regions in videos.
    
    This class implements a super-resolution pipeline specifically optimized
    for facial features, with support for 2x and 4x upscaling.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the FaceSuperResolution model.
        
        Args:
            model_path: Path to pre-trained model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        super().__init__()
        self.device = device
        self.upscale_factor = 4  # Default upscale factor
        self.face_detector = self._init_face_detector()
        self.model = self._load_model(model_path)
        self.face_enhancer = None  # Can be initialized on-demand
        
        # Initialize image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _init_face_detector(self):
        """Initialize face detection model."""
        try:
            # Using OpenCV's DNN face detector as a lightweight option
            proto_path = str(Path(__file__).parent / 'weights' / 'deploy.prototxt')
            model_path = str(Path(__file__).parent / 'weights' / 'res10_300x300_ssd_iter_140000.caffemodel')
            
            if not Path(proto_path).exists() or not Path(model_path).exists():
                logger.warning("Face detection model files not found. Face detection will be disabled.")
                return None
                
            net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            if 'cuda' in self.device:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            return net
        except Exception as e:
            logger.warning(f"Failed to initialize face detector: {e}")
            return None
    
    def _load_model(self, model_path: Optional[str] = None):
        """
        Load the Real-ESRGAN model for super-resolution.
        
        Args:
            model_path: Optional path to custom model weights
            
        Returns:
            Loaded Real-ESRGAN model
        """
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            logger.info("Initializing Real-ESRGAN model...")
            
            # Determine model name and scale based on upscale factor
            if self.upscale_factor == 2:
                model_name = 'RealESRGAN_x2plus'
                scale = 2
            else:  # 4x upscale
                model_name = 'RealESRGAN_x4plus'
                scale = 4
            
            # Initialize the model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=scale
            )
            
            # Initialize the upscaler
            upsampler = RealESRGANer(
                scale=scale,
                model_path=model_path,
                model=model,
                tile=400,  # Tile size for processing
                tile_pad=10,
                pre_pad=0,
                half=True if 'cuda' in self.device else False
            )
            
            logger.info(f"Loaded {model_name} model")
            return upsampler
            
        except ImportError as e:
            logger.error(f"Failed to import Real-ESRGAN: {e}")
            logger.warning("Falling back to basic super-resolution")
            return None
    
    def detect_faces(self, image: np.ndarray) -> list:
        """
        Detect faces in the input image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of face bounding boxes in (x, y, w, h) format
        """
        if self.face_detector is None:
            # Fallback to full image if face detection is not available
            h, w = image.shape[:2]
            return [(0, 0, w, h)]
            
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure the bounding boxes fall within the dimensions of the image
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)
                
                # Add some padding around the face
                padding = 20
                startX = max(0, startX - padding)
                startY = max(0, startY - padding)
                endX = min(w, endX + padding)
                endY = min(h, endY + padding)
                
                faces.append((startX, startY, endX - startX, endY - startY))
        
        return faces if faces else [(0, 0, w, h)]
    
    def enhance_face(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Enhance a face region using Real-ESRGAN super-resolution.
        
        Args:
            image: Input image in BGR format
            face_box: Face bounding box in (x, y, w, h) format
            
        Returns:
            Enhanced face region in BGR format
        """
        if self.model is None:
            logger.warning("Super-resolution model not loaded. Returning original image.")
            return image
            
        x, y, w, h = face_box
        face_region = image[y:y+h, x:x+w].copy()
        
        try:
            # Convert to RGB for Real-ESRGAN
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Apply super-resolution
            output, _ = self.model.enhance(
                face_rgb,
                outscale=self.upscale_factor,
                alpha_upsampler='realesrgan',
                tile=400,
                tile_pad=10,
                pre_pad=0
            )
            
            # Convert back to BGR for OpenCV
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # Resize back to original dimensions if needed
            if output.shape[0] != h or output.shape[1] != w:
                output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LANCZOS4)
                
            return output
            
        except Exception as e:
            logger.error(f"Error in face enhancement: {e}")
            return face_region
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame to enhance face regions.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Enhanced frame with super-resolved face regions
        """
        if self.model is None:
            return frame
            
        # Create a copy of the frame to avoid modifying the original
        enhanced_frame = frame.copy()
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # Skip very small face detections
            if w < 50 or h < 50:
                continue
                
            # Enhance the face region
            enhanced_face = self.enhance_face(frame, (x, y, w, h))
            
            # Resize enhanced face to match the original face region size
            enhanced_face = cv2.resize(enhanced_face, (w, h))
            
            # Blend the enhanced face back into the frame
            mask = np.ones(enhanced_face.shape, dtype=enhanced_face.dtype) * 255
            center = (x + w // 2, y + h // 2)
            
            # Use seamless cloning to blend the enhanced face
            try:
                enhanced_frame = cv2.seamlessClone(
                    enhanced_face, enhanced_frame, mask, center, cv2.NORMAL_CLONE
                )
            except Exception as e:
                logger.warning(f"Error in seamless cloning: {e}")
                # Fallback to simple replacement if seamless cloning fails
                enhanced_frame[y:y+h, x:x+w] = enhanced_face
        
        return enhanced_frame
    
    def enhance_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_resolution: Tuple[int, int] = (1920, 1080),
        fps: Optional[float] = None,
        batch_size: int = 4
    ) -> str:
        """
        Enhance a video by applying super-resolution to face regions.
        
        Args:
            input_path: Path to the input video file
            output_path: Path to save the enhanced video
            target_resolution: Target resolution as (width, height)
            fps: Frames per second for the output video (uses input FPS if None)
            batch_size: Number of frames to process in parallel
            
        Returns:
            Path to the enhanced video file
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract frames from the input video
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
        
        # Process frames in batches
        enhanced_frames = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            
            for frame_path in batch:
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
        
        # Combine processed frames into a video
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
    
    def set_upscale_factor(self, factor: int) -> None:
        """
        Set the upscaling factor for super-resolution.
        
        Args:
            factor: Upscaling factor (2 or 4)
        """
        if factor not in [2, 4]:
            raise ValueError("Upscale factor must be either 2 or 4")
        self.upscale_factor = factor
        logger.info(f"Upscale factor set to {factor}x")


class BackgroundUpscaler:
    """
    Background upscaler for enhancing non-face regions of the video.
    Uses a lightweight upscaling approach to improve background quality.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the background upscaler.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = self._load_model()
    
    def _load_model(self):
        """Load the background upscaling model."""
        # Placeholder for model loading logic
        logger.info("Initializing background upscaler...")
        return None
    
    def upscale_background(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Upscale the background region of a frame.
        
        Args:
            frame: Input frame in BGR format
            mask: Binary mask where 255 indicates foreground (to be preserved)
                  and 0 indicates background (to be upscaled)
                  
        Returns:
            Frame with upscaled background
        """
        if self.model is None:
            return frame
            
        # Invert mask to get background regions
        bg_mask = 255 - mask
        
        # Apply Gaussian blur to the mask for smoother blending
        bg_mask = cv2.GaussianBlur(bg_mask, (21, 21), 0) / 255.0
        fg_mask = 1.0 - bg_mask
        
        # Convert masks to 3 channels
        bg_mask = cv2.merge([bg_mask, bg_mask, bg_mask])
        fg_mask = cv2.merge([fg_mask, fg_mask, fg_mask])
        
        # Apply upscaling to the background
        # In a real implementation, this would use the loaded model
        upscaled_bg = frame.copy()
        
        # Blend the upscaled background with the original frame
        result = frame * fg_mask + upscaled_bg * bg_mask
        
        return result.astype(np.uint8)


def create_super_resolution_pipeline(
    face_model_path: Optional[str] = None,
    bg_model_path: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[FaceSuperResolution, BackgroundUpscaler]:
    """
    Create a super-resolution pipeline with face and background enhancement.
    
    Args:
        face_model_path: Path to face super-resolution model weights
        bg_model_path: Path to background upscaling model weights
        device: Device to run models on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (face_enhancer, bg_upscaler)
    """
    face_enhancer = FaceSuperResolution(model_path=face_model_path, device=device)
    bg_upscaler = BackgroundUpscaler(device=device)
    
    return face_enhancer, bg_upscaler
