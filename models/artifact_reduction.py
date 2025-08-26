"""
Artifact Reduction Module

This module provides functionality for reducing compression artifacts and noise
in videos and images, with a focus on preserving important details while removing
unwanted visual artifacts from compression, upscaling, or other processing steps.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from pathlib import Path
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)

class ArtifactReductionNetwork(nn.Module):
    """
    Neural network for artifact reduction using a U-Net like architecture.
    This is a simplified version focusing on artifact reduction.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        
        # Middle
        self.middle = self._conv_block(256, 512)
        
        # Decoder
        self.dec3 = self._up_conv_block(512, 256)
        self.dec2 = self._up_conv_block(512, 128)
        self.dec1 = self._up_conv_block(256, 64)
        
        # Output
        self.out_conv = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        
        # Initialize weights
        self._init_weights()
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block with batch normalization and LeakyReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)
        )
    
    def _up_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create an upsampling block with transposed convolution."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Middle
        m = self.middle(e3)
        
        # Decoder with skip connections
        d3 = self.dec3(m)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        # Output
        out = torch.sigmoid(self.out_conv(d1))
        return out


class ArtifactReducer(BaseModel):
    """
    Artifact reduction module that combines traditional image processing
    with deep learning for high-quality artifact reduction.
    """
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        model_path: Optional[str] = None
    ):
        """
        Initialize the ArtifactReducer.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            model_path: Path to pre-trained model weights
        """
        super().__init__()
        self.device = device
        self.model = self._load_model(model_path)
        self._init_image_processing()
    
    def _load_model(self, model_path: Optional[str] = None) -> Optional[nn.Module]:
        """Load the artifact reduction model."""
        try:
            model = ArtifactReductionNetwork()
            
            if model_path and Path(model_path).exists():
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
            
            model = model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            logger.warning(f"Failed to load artifact reduction model: {e}")
            return None
    
    def _init_image_processing(self):
        """Initialize parameters for traditional image processing."""
        # Bilateral filter parameters
        self.bilateral_d = 5  # Diameter of each pixel neighborhood
        self.bilateral_sigma_color = 75  # Filter sigma in the color space
        self.bilateral_sigma_space = 75  # Filter sigma in the coordinate space
        
        # Non-local means denoising parameters
        self.nl_means_h = 10  # h parameter for NLM, higher preserves more detail but removes less noise
        self.nl_means_template_window = 7  # Size in pixels of the template patch
        self.nl_means_search_window = 21  # Size in pixels of the window that is used to compute weighted average
        
        # Edge-aware smoothing parameters
        self.edge_guided_filter_radius = 5
        self.edge_guided_filter_eps = 0.1
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for the neural network."""
        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor
    
    def _postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert model output back to numpy image."""
        # Remove batch dimension and permute back to HWC
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Convert back to uint8
        image = (image * 255).clip(0, 255).astype(np.uint8)
        return image
    
    def _apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filter to reduce noise while preserving edges."""
        return cv2.bilateralFilter(
            image,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
    
    def _apply_nl_means(self, image: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising."""
        return cv2.fastNlMeansDenoisingColored(
            image,
            h=self.nl_means_h,
            templateWindowSize=self.nl_means_template_window,
            searchWindowSize=self.nl_means_search_window
        )
    
    def _apply_edge_guided_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply edge-guided filtering."""
        # Convert to float32 for processing
        image_float = image.astype(np.float32) / 255.0
        
        # Apply guided filter
        guided = cv2.ximgproc.guidedFilter(
            guide=image_float,
            src=image_float,
            radius=self.edge_guided_filter_radius,
            eps=self.edge_guided_filter_eps
        )
        
        # Convert back to uint8
        return (guided * 255).clip(0, 255).astype(np.uint8)
    
    def reduce_artifacts_deep(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce artifacts using the deep learning model.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Processed image with reduced artifacts
        """
        if self.model is None:
            logger.warning("Deep learning model not available, falling back to traditional methods")
            return self.reduce_artifacts_traditional(image)
        
        try:
            with torch.no_grad():
                # Preprocess
                input_tensor = self._preprocess_image(image)
                
                # Forward pass
                output_tensor = self.model(input_tensor)
                
                # Postprocess
                result = self._postprocess_image(output_tensor)
                return result
                
        except Exception as e:
            logger.error(f"Error in deep artifact reduction: {e}")
            return image
    
    def reduce_artifacts_traditional(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce artifacts using traditional image processing techniques.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Processed image with reduced artifacts
        """
        try:
            # Apply non-local means denoising
            denoised = self._apply_nl_means(image)
            
            # Apply bilateral filter
            filtered = self._apply_bilateral_filter(denoised)
            
            # Apply edge-guided filtering
            result = self._apply_edge_guided_filter(filtered)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in traditional artifact reduction: {e}")
            return image
    
    def process_frame(self, frame: np.ndarray, use_deep: bool = True) -> np.ndarray:
        """
        Process a single frame to reduce artifacts.
        
        Args:
            frame: Input frame in BGR format
            use_deep: Whether to use deep learning model if available
            
        Returns:
            Processed frame with reduced artifacts
        """
        if use_deep and self.model is not None:
            return self.reduce_artifacts_deep(frame)
        return self.reduce_artifacts_traditional(frame)
    
    def enhance_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_resolution: Optional[Tuple[int, int]] = None,
        fps: Optional[float] = None,
        use_deep: bool = True
    ) -> str:
        """
        Process a video to reduce artifacts.
        
        Args:
            input_path: Path to the input video file
            output_path: Path to save the processed video
            target_resolution: Target resolution as (width, height)
            fps: Frames per second for the output video
            use_deep: Whether to use deep learning model if available
            
        Returns:
            Path to the processed video file
        """
        from ..utils.video_utils import extract_frames, combine_frames_to_video
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract frames
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
        processed_frames = []
        for i, frame_path in enumerate(frames):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"Failed to read frame: {frame_path}")
                continue
            
            # Process the frame
            processed_frame = self.process_frame(frame, use_deep=use_deep)
            
            # Resize to target resolution if specified
            if target_resolution and (frame.shape[1], frame.shape[0]) != target_resolution:
                processed_frame = cv2.resize(
                    processed_frame,
                    target_resolution,
                    interpolation=cv2.INTER_LANCZOS4
                )
            
            processed_frames.append(processed_frame)
            
            # Log progress
            if (i + 1) % 10 == 0 or (i + 1) == len(frames):
                logger.info(f"Processed {i + 1}/{len(frames)} frames")
        
        # Combine frames into video
        if not processed_frames:
            raise RuntimeError("No frames were processed successfully")
        
        # Use the first frame to get video properties
        height, width = processed_frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps or 30.0,  # Default to 30 FPS if not specified
            (width, height)
        )
        
        # Write frames to video
        for frame in processed_frames:
            out.write(frame)
        
        # Release resources
        out.release()
        
        return str(output_path)


def create_artifact_reducer(
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    model_path: Optional[str] = None,
    use_deep: bool = True
) -> ArtifactReducer:
    """
    Create an artifact reducer with the specified parameters.
    
    Args:
        device: Device to run the model on ('cuda' or 'cpu')
        model_path: Path to pre-trained model weights
        use_deep: Whether to use deep learning model if available
        
    Returns:
        Configured ArtifactReducer instance
    """
    reducer = ArtifactReducer(device=device, model_path=model_path)
    if not use_deep:
        reducer.model = None  # Force traditional methods only
    return reducer
