""Video processing utilities for PaksaTalker."""
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path
import subprocess
import shutil

from config import config

def extract_frames(
    video_path: str, 
    output_dir: str, 
    fps: Optional[float] = None,
    max_frames: Optional[int] = None
) -> List[str]:
    """Extract frames from a video file.
    
    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        fps: Frames per second to extract. If None, extracts all frames.
        max_frames: Maximum number of frames to extract. If None, extracts all frames.
        
    Returns:
        List of paths to extracted frame images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use OpenCV to extract frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval for target FPS
    frame_interval = 1.0
    if fps is not None and fps < original_fps:
        frame_interval = original_fps / fps
    
    # Process frames
    frame_paths = []
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if we should save this frame based on FPS
        if frame_count % int(round(frame_interval)) == 0:
            # Check if we've reached the maximum number of frames
            if max_frames is not None and saved_count >= max_frames:
                break
                
            # Save frame as image
            frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return frame_paths

def create_video(
    frame_paths: List[str],
    output_path: str,
    fps: float = 30.0,
    audio_path: Optional[str] = None,
    overwrite: bool = True
) -> str:
    """Create a video from a sequence of frames.
    
    Args:
        frame_paths: List of paths to frame images.
        output_path: Path to save the output video.
        fps: Frames per second for the output video.
        audio_path: Optional path to audio file to include in the video.
        overwrite: Whether to overwrite existing output file.
        
    Returns:
        Path to the created video file.
    """
    if not frame_paths:
        raise ValueError("No frame paths provided")
    
    # Get frame dimensions from first frame
    frame = cv2.imread(frame_paths[0])
    if frame is None:
        raise ValueError(f"Could not read frame: {frame_paths[0]}")
    
    height, width = frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}, skipping")
            continue
        out.write(frame)
    
    out.release()
    
    # Add audio if provided
    if audio_path and os.path.exists(audio_path):
        temp_output = f"{output_path}.temp.mp4"
        
        # Use ffmpeg to add audio
        cmd = [
            'ffmpeg',
            '-y' if overwrite else '-n',
            '-i', output_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            temp_output
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            shutil.move(temp_output, output_path)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to add audio: {e.stderr.decode()}")
            if os.path.exists(temp_output):
                os.remove(temp_output)
    
    return output_path

def resize_video(
    input_path: str,
    output_path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: float = 1.0,
    keep_aspect_ratio: bool = True
) -> str:
    """Resize a video to the specified dimensions.
    
    Args:
        input_path: Path to the input video file.
        output_path: Path to save the resized video.
        width: Target width in pixels. If None, scales based on height.
        height: Target height in pixels. If None, scales based on width.
        scale: Scale factor for resizing. Only used if width and height are None.
        keep_aspect_ratio: Whether to maintain the aspect ratio when resizing.
        
    Returns:
        Path to the resized video file.
    """
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate target dimensions
    if width is None and height is None:
        width = int(orig_width * scale)
        height = int(orig_height * scale)
    elif keep_aspect_ratio:
        if width is not None and height is None:
            # Calculate height based on width to maintain aspect ratio
            aspect_ratio = orig_width / orig_height
            height = int(width / aspect_ratio)
        elif height is not None and width is None:
            # Calculate width based on height to maintain aspect ratio
            aspect_ratio = orig_width / orig_height
            width = int(height * aspect_ratio)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        out.write(resized_frame)
    
    # Release resources
    cap.release()
    out.release()
    
    return output_path

def extract_audio(
    video_path: str,
    output_path: str,
    overwrite: bool = True
) -> str:
    """Extract audio from a video file.
    
    Args:
        video_path: Path to the input video file.
        output_path: Path to save the extracted audio.
        overwrite: Whether to overwrite existing output file.
        
    Returns:
        Path to the extracted audio file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Use ffmpeg to extract audio
    cmd = [
        'ffmpeg',
        '-y' if overwrite else '-n',
        '-i', video_path,
        '-q:a', '0',
        '-map', 'a',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")

def get_video_info(video_path: str) -> dict:
    """Get information about a video file.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        Dictionary containing video information.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        'path': video_path,
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': float(cap.get(cv2.CAP_PROP_FPS)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / float(cap.get(cv2.CAP_PROP_FPS)),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, byteorder=sys.byteorder).decode('ascii')
    }
    
    cap.release()
    return info
