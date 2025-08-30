import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
import tempfile

def extract_frames(video_path: str, output_dir: str, fps: int = 25) -> Tuple[List[str], float]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract
        
    Returns:
        Tuple of (list of frame paths, video duration in seconds)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / original_fps
    
    # Calculate frame interval based on target FPS
    frame_interval = max(1, int(original_fps / fps)) if fps < original_fps else 1
    
    frame_paths = []
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame at specified interval
        if frame_idx % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
            
        frame_idx += 1
    
    cap.release()
    return frame_paths, duration

def process_uploaded_file(file_path: str, output_dir: str) -> Tuple[List[str], str, bool]:
    """
    Process an uploaded file (image or video) and return frame paths.
    
    Args:
        file_path: Path to the uploaded file
        output_dir: Directory to save processed frames
        
    Returns:
        Tuple of (list of frame paths, media type, is_video)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check file extension to determine type
    ext = os.path.splitext(file_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    if is_video:
        # For videos, extract frames
        frame_paths, _ = extract_frames(file_path, output_dir)
        return frame_paths, 'video', True
    else:
        # For images, just return the single frame path
        return [file_path], 'image', False

def create_video_from_frames(frame_paths: List[str], output_path: str, fps: int = 25) -> str:
    """
    Create a video from a list of frame paths.
    
    Args:
        frame_paths: List of paths to image frames
        output_path: Path to save the output video
        fps: Frames per second for the output video
        
    Returns:
        Path to the created video file
    """
    if not frame_paths:
        raise ValueError("No frames provided to create video")
    
    # Read first frame to get dimensions
    frame = cv2.imread(frame_paths[0])
    if frame is None:
        raise ValueError(f"Could not read frame: {frame_paths[0]}")
    
    height, width = frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame_path in tqdm(frame_paths, desc="Creating video"):
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
    
    out.release()
    return output_path
