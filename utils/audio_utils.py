""Audio processing utilities for PaksaTalker."""
import os
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple, Union, List
import subprocess
import tempfile
import shutil

from config import config

def load_audio(
    file_path: str,
    sr: Optional[int] = None,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
    resample: bool = True
) -> Tuple[np.ndarray, int]:
    """Load an audio file.
    
    Args:
        file_path: Path to the audio file.
        sr: Target sample rate. If None, uses the file's original sample rate.
        mono: Whether to convert to mono.
        duration: Maximum duration in seconds to load.
        offset: Start reading after this time (in seconds).
        resample: Whether to resample the audio to the target sample rate.
        
    Returns:
        A tuple containing the audio data as a numpy array and the sample rate.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Load audio with librosa
    y, sr_orig = librosa.load(
        file_path,
        sr=sr if resample else None,
        mono=mono,
        duration=duration,
        offset=offset,
        res_type='kaiser_fast'
    )
    
    # Ensure audio is at least 1D
    if len(y.shape) == 1:
        y = y.reshape(1, -1)
    
    return y, sr_orig if sr is None or not resample else sr

def save_audio(
    y: np.ndarray,
    file_path: str,
    sr: int = 22050,
    format: Optional[str] = None,
    subtype: Optional[str] = None
) -> None:
    """Save an audio signal to a file.
    
    Args:
        y: Audio data as a numpy array (shape: [channels, samples] or [samples] for mono).
        file_path: Path to save the audio file.
        sr: Sample rate of the audio.
        format: Audio format (e.g., 'WAV', 'MP3'). If None, inferred from file extension.
        subtype: Subtype of the audio format. If None, uses a default for the format.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)) or '.', exist_ok=True)
    
    # Convert to correct shape if needed
    if len(y.shape) == 1:
        y = y.reshape(1, -1)  # Make it 2D: [1, samples]
    
    # Save using soundfile
    sf.write(file_path, y.T, sr, format=format, subtype=subtype)

def resample_audio(
    y: np.ndarray,
    orig_sr: int,
    target_sr: int,
    res_type: str = 'kaiser_fast'
) -> np.ndarray:
    """Resample an audio signal to a target sample rate.
    
    Args:
        y: Audio data as a numpy array.
        orig_sr: Original sample rate of the audio.
        target_sr: Target sample rate.
        res_type: Resampling method.
        
    Returns:
        Resampled audio data.
    """
    if orig_sr == target_sr:
        return y
    
    return librosa.resample(
        y,
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type=res_type
    )

def trim_silence(
    y: np.ndarray,
    sr: int,
    top_db: float = 30,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """Trim leading and trailing silence from an audio signal.
    
    Args:
        y: Audio data as a numpy array.
        sr: Sample rate of the audio.
        top_db: The threshold (in decibels) below reference to consider as silence.
        frame_length: Number of samples per analysis frame.
        hop_length: Number of samples between frames.
        
    Returns:
        Trimmed audio data.
    """
    if len(y.shape) > 1:  # Multi-channel
        return np.vstack([
            librosa.effects.trim(channel, top_db=top_db, frame_length=frame_length, hop_length=hop_length)[0]
            for channel in y
        ])
    else:  # Mono
        return librosa.effects.trim(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)[0]

def normalize_audio(
    y: np.ndarray,
    target_db: float = -3.0,
    max_gain_db: float = 30.0
) -> np.ndarray:
    """Normalize audio to a target level.
    
    Args:
        y: Audio data as a numpy array.
        target_db: Target level in decibels.
        max_gain_db: Maximum gain in decibels to prevent excessive amplification of noise.
        
    Returns:
        Normalized audio data.
    """
    # Calculate RMS and convert to dB
    rms = np.sqrt(np.mean(y**2))
    if rms < 1e-6:  # Avoid division by zero
        return y
    
    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    
    # Limit the gain to prevent excessive amplification of noise
    gain_db = max(min(gain_db, max_gain_db), -max_gain_db)
    
    # Apply gain
    return y * (10 ** (gain_db / 20.0))

def mix_audios(
    audio_paths: List[str],
    output_path: str,
    sr: int = 22050,
    normalize: bool = True
) -> str:
    """Mix multiple audio files together.
    
    Args:
        audio_paths: List of paths to audio files.
        output_path: Path to save the mixed audio.
        sr: Target sample rate for the output.
        normalize: Whether to normalize the mixed audio.
        
    Returns:
        Path to the mixed audio file.
    """
    if not audio_paths:
        raise ValueError("No audio files provided")
    
    # Load all audio files
    audios = []
    max_length = 0
    
    for path in audio_paths:
        if not os.path.exists(path):
            print(f"Warning: Audio file not found: {path}")
            continue
            
        try:
            y, orig_sr = load_audio(path, sr=sr, mono=True)
            
            # Reshape to 2D if needed
            if len(y.shape) == 1:
                y = y.reshape(1, -1)
                
            audios.append(y)
            max_length = max(max_length, y.shape[1])
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    
    if not audios:
        raise ValueError("No valid audio files to mix")
    
    # Mix audio
    mixed = np.zeros((1, max_length))
    
    for audio in audios:
        # Pad or truncate to match max length
        if audio.shape[1] < max_length:
            # Pad with zeros
            pad_width = ((0, 0), (0, max_length - audio.shape[1]))
            audio = np.pad(audio, pad_width, mode='constant')
        elif audio.shape[1] > max_length:
            # Truncate
            audio = audio[:, :max_length]
            
        mixed += audio
    
    # Normalize if needed
    if normalize:
        mixed = normalize_audio(mixed)
    
    # Save the mixed audio
    save_audio(mixed, output_path, sr=sr)
    
    return output_path

def convert_audio_format(
    input_path: str,
    output_path: str,
    format: Optional[str] = None,
    sr: Optional[int] = None,
    channels: Optional[int] = None,
    bitrate: Optional[str] = None,
    overwrite: bool = True
) -> str:
    """Convert an audio file to a different format.
    
    Args:
        input_path: Path to the input audio file.
        output_path: Path to save the converted audio.
        format: Output format (e.g., 'wav', 'mp3'). If None, inferred from output_path.
        sr: Target sample rate. If None, keeps original sample rate.
        channels: Number of output channels. If None, keeps original channels.
        bitrate: Audio bitrate (e.g., '192k' for 192 kbps).
        overwrite: Whether to overwrite existing output file.
        
    Returns:
        Path to the converted audio file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    
    # Build ffmpeg command
    cmd = ['ffmpeg']
    
    # Overwrite output file if it exists
    if overwrite:
        cmd.append('-y')
    else:
        cmd.append('-n')
    
    # Input file
    cmd.extend(['-i', input_path])
    
    # Set output format if specified
    if format is not None:
        cmd.extend(['-f', format])
    
    # Set sample rate if specified
    if sr is not None:
        cmd.extend(['-ar', str(sr)])
    
    # Set number of channels if specified
    if channels is not None:
        cmd.extend(['-ac', str(channels)])
    
    # Set bitrate if specified
    if bitrate is not None:
        cmd.extend(['-b:a', bitrate])
    
    # Output file
    cmd.append(output_path)
    
    # Run ffmpeg
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert audio: {e.stderr.decode()}")

def get_audio_info(file_path: str) -> dict:
    """Get information about an audio file.
    
    Args:
        file_path: Path to the audio file.
        
    Returns:
        Dictionary containing audio information.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Get basic info with soundfile
        info = {}
        with sf.SoundFile(file_path) as f:
            info.update({
                'path': file_path,
                'channels': f.channels,
                'samplerate': f.samplerate,
                'duration': float(f.frames) / f.samplerate,
                'format': f.format,
                'subtype': f.subtype,
                'frames': f.frames,
            })
        
        # Get additional info with librosa
        y, sr = librosa.load(file_path, sr=None, mono=False)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)  # Ensure 2D
        
        # Calculate RMS and peak levels
        rms = np.sqrt(np.mean(y**2, axis=1))
        peak = np.max(np.abs(y), axis=1)
        
        info.update({
            'rms_db': 20 * np.log10(np.maximum(rms, 1e-10)).tolist(),
            'peak_db': 20 * np.log10(np.maximum(peak, 1e-10)).tolist(),
            'is_silent': np.all(rms < 1e-6),
        })
        
        return info
        
    except Exception as e:
        raise RuntimeError(f"Failed to get audio info: {e}")
