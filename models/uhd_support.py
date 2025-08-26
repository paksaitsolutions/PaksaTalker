"""
4K/UHD Support Module

This module provides functionality for processing and generating 4K (3840x2160) video content
with optimizations for memory usage and performance.
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple, Union, List, Callable
from pathlib import Path
import logging
from tqdm import tqdm
import os
import gc

from .base import BaseModel
from .super_resolution import FaceSuperResolution
from .background_upscaler import BackgroundUpscaler
from .artifact_reduction import ArtifactReducer

logger = logging.getLogger(__name__)

class UHDProcessor:
    """
    Handles 4K/UHD video processing with optimizations for memory and performance.
    """
    
    # Standard 4K resolution
    UHD_RESOLUTION = (3840, 2160)
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: int = 1,
        use_fp16: bool = True
    ):
        """
        Initialize the UHD processor.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            tile_size: Size of tiles for tiled processing
            overlap: Overlap between tiles to avoid seams
            batch_size: Number of tiles to process in parallel
            use_fp16: Whether to use mixed precision (FP16) for faster processing
        """
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and ('cuda' in device)  # FP16 only on CUDA
        
        # Initialize submodules
        self.face_sr = FaceSuperResolution(device=device)
        self.bg_upscaler = BackgroundUpscaler(device=device)
        self.artifact_reducer = ArtifactReducer(device=device)
        
        # Memory management
        self.max_memory_gb = self._get_available_memory() * 0.7  # Use 70% of available memory
        
        logger.info(f"Initialized UHD processor with {self.max_memory_gb:.2f}GB memory budget")
    
    def _get_available_memory(self) -> float:
        """Get available GPU memory in GB or system RAM if CUDA is not available."""
        if 'cuda' in self.device and torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        
        # Fallback to system RAM
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    
    def _estimate_memory_usage(self, width: int, height: int) -> float:
        """
        Estimate memory usage for processing a frame.
        
        Returns:
            Estimated memory usage in GB
        """
        # Base memory for the frame (4 bytes per pixel for float32)
        frame_mem = (width * height * 3 * 4) / (1024 ** 3)
        
        # Memory for intermediate tensors (approximate)
        intermediate_mem = frame_mem * 4  # Conservative estimate
        
        return frame_mem + intermediate_mem
    
    def _split_into_tiles(
        self, 
        image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Split image into overlapping tiles for processing.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Tuple of (tiles, positions) where positions are (x1, y1, x2, y2)
        """
        h, w = image.shape[:2]
        tiles = []
        positions = []
        
        # Calculate step size
        step = self.tile_size - 2 * self.overlap
        
        # Adjust step size to cover the entire image
        if step <= 0:
            raise ValueError("Tile size must be greater than twice the overlap")
        
        # Generate tiles
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Calculate tile boundaries with overlap
                x1 = max(0, x - self.overlap)
                y1 = max(0, y - self.overlap)
                x2 = min(w, x + self.tile_size + self.overlap)
                y2 = min(h, y + self.tile_size + self.overlap)
                
                # Extract tile
                tile = image[y1:y2, x1:x2]
                tiles.append(tile)
                positions.append((x1, y1, x2, y2))
        
        return tiles, positions
    
    def _merge_tiles(
        self, 
        tiles: List[np.ndarray], 
        positions: List[Tuple[int, int, int, int]],
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Merge processed tiles back into a single image using weighted blending.
        
        Args:
            tiles: List of processed tiles
            positions: List of tile positions (x1, y1, x2, y2)
            target_shape: Shape of the output image (height, width, channels)
            
        Returns:
            Merged image
        """
        result = np.zeros(target_shape, dtype=np.float32)
        weight = np.zeros(target_shape[:2], dtype=np.float32)
        
        for tile, (x1, y1, x2, y2) in zip(tiles, positions):
            th, tw = tile.shape[:2]
            
            # Create weight mask (cosine window)
            mask = np.ones((th, tw), dtype=np.float32)
            
            # Apply cosine window to edges to reduce seams
            window_size = self.overlap
            if window_size > 0:
                # Create cosine window
                window = np.ones_like(mask)
                
                # Apply to left and right edges
                if x1 > 0:  # Not the first column
                    window[:, :window_size] = np.hanning(2 * window_size)[:window_size].reshape(1, -1)
                if x2 < target_shape[1]:  # Not the last column
                    window[:, -window_size:] = np.hanning(2 * window_size)[-window_size:].reshape(1, -1)
                
                # Apply to top and bottom edges
                if y1 > 0:  # Not the first row
                    window[:window_size, :] *= np.hanning(2 * window_size)[:window_size].reshape(-1, 1)
                if y2 < target_shape[0]:  # Not the last row
                    window[-window_size:, :] *= np.hanning(2 * window_size)[-window_size:].reshape(-1, 1)
                
                mask = window
            
            # Add weighted tile to result
            result[y1:y2, x1:x2] += tile * mask[..., None]
            weight[y1:y2, x1:x2] += mask
        
        # Normalize by weight to blend overlapping regions
        result = np.divide(result, weight[..., None], where=weight[..., None] > 0)
        return result.astype(np.uint8)
    
    def _process_tile(self, tile: np.ndarray) -> np.ndarray:
        """Process a single tile with the enhancement pipeline."""
        # Convert to float32 for processing
        tile = tile.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Convert to FP16 if enabled
        if self.use_fp16:
            tensor = tensor.half()
        
        # Apply enhancement pipeline
        with torch.no_grad():
            # Face super-resolution
            tensor = self.face_sr.model(tensor) if self.face_sr.model is not None else tensor
            
            # Convert back to float32 for further processing
            tensor = tensor.float()
            
            # Background upscaling
            if self.bg_upscaler.model is not None:
                tensor = self.bg_upscaler.model(tensor)
            
            # Artifact reduction
            if self.artifact_reducer.model is not None:
                tensor = self.artifact_reducer.model(tensor)
        
        # Convert back to numpy
        result = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        
        return result
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with 4K enhancement.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Enhanced frame in BGR format
        """
        original_dtype = frame.dtype
        h, w = frame.shape[:2]
        
        # Check if tiling is needed
        mem_required = self._estimate_memory_usage(w, h)
        use_tiling = mem_required > (self.max_memory_gb * 0.5)  # Use tiling if > 50% of memory
        
        if not use_tiling:
            # Process entire frame at once if it fits in memory
            return self._process_tile(frame)
        
        # Split into tiles
        tiles, positions = self._split_into_tiles(frame)
        processed_tiles = []
        
        # Process tiles in batches
        for i in range(0, len(tiles), self.batch_size):
            batch = tiles[i:i + self.batch_size]
            
            # Process batch
            with torch.no_grad():
                processed_batch = [self._process_tile(tile) for tile in batch]
                processed_tiles.extend(processed_batch)
            
            # Clear CUDA cache to free memory
            if 'cuda' in self.device:
                torch.cuda.empty_cache()
        
        # Merge tiles back together
        result = self._merge_tiles(processed_tiles, positions, frame.shape)
        
        # Clean up
        del tiles, processed_tiles
        gc.collect()
        
        return result.astype(original_dtype)
    
    def process_video(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_resolution: Tuple[int, int] = UHD_RESOLUTION,
        fps: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """
        Process a video with 4K enhancement.
        
        Args:
            input_path: Path to the input video file
            output_path: Path to save the processed video
            target_resolution: Target resolution as (width, height)
            fps: Frames per second for the output video
            progress_callback: Optional callback for progress updates (0.0 to 1.0)
            
        Returns:
            Path to the processed video file
        """
        from ..utils.video_utils import get_video_info, VideoReader, VideoWriter
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get video info
        video_info = get_video_info(str(input_path))
        if not video_info:
            raise ValueError(f"Could not read video info: {input_path}")
        
        # Use input FPS if not specified
        if fps is None:
            fps = video_info['fps']
        
        # Initialize video reader and writer
        reader = VideoReader(str(input_path))
        writer = VideoWriter(
            str(output_path),
            codec='libx264',
            fps=fps,
            resolution=target_resolution,
            preset='slow',
            crf=18
        )
        
        try:
            # Process frames
            total_frames = video_info['frame_count']
            processed_frames = 0
            
            with tqdm(total=total_frames, desc="Processing video") as pbar:
                for frame in reader:
                    if frame is None:
                        break
                    
                    # Process frame
                    processed_frame = self.process_frame(frame)
                    
                    # Resize to target resolution if needed
                    if (processed_frame.shape[1], processed_frame.shape[0]) != target_resolution:
                        processed_frame = cv2.resize(
                            processed_frame,
                            target_resolution,
                            interpolation=cv2.INTER_LANCZOS4
                        )
                    
                    # Write frame
                    writer.write(processed_frame)
                    
                    # Update progress
                    processed_frames += 1
                    pbar.update(1)
                    
                    if progress_callback:
                        progress = processed_frames / total_frames
                        progress_callback(progress)
                    
                    # Clear memory
                    del frame, processed_frame
                    if 'cuda' in self.device:
                        torch.cuda.empty_cache()
        
        finally:
            # Clean up
            reader.release()
            writer.release()
        
        return str(output_path)


def create_uhd_processor(
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 1,
    use_fp16: bool = True
) -> UHDProcessor:
    """
    Create a UHD processor with the specified parameters.
    
    Args:
        device: Device to run the model on ('cuda' or 'cpu')
        tile_size: Size of tiles for tiled processing
        overlap: Overlap between tiles to avoid seams
        batch_size: Number of tiles to process in parallel
        use_fp16: Whether to use mixed precision (FP16) for faster processing
        
    Returns:
        Configured UHDProcessor instance
    """
    return UHDProcessor(
        device=device,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        use_fp16=use_fp16
    )
