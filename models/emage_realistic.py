"""
EMAGE - Expressive Motion Avatar Generation
Best realistic body expression model implementation
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
import subprocess
import logging
from typing import Optional, Dict, Any, List
import cv2

logger = logging.getLogger(__name__)

class EMageRealistic:
    """EMAGE model for realistic body expressions and gestures"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.model = None
        self.audio_encoder = None
        self.motion_decoder = None
        self.loaded = False
        
    def _get_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self):
        """Load EMAGE model from official repository"""
        try:
            logger.info("Loading EMAGE realistic body expression model...")
            
            # Clone EMAGE repository
            repo_path = Path("EMAGE")
            if not repo_path.exists():
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/PantoMatrix/EMAGE.git"
                ], check=True)
            
            # Install requirements if exists
            requirements_file = repo_path / "requirements.txt"
            if requirements_file.exists():
                try:
                    subprocess.run([
                        "pip", "install", "-r", 
                        str(requirements_file)
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install EMAGE requirements: {e}")
            else:
                logger.info("EMAGE requirements.txt not found, continuing without installation")
            
            # Add to Python path
            import sys
            sys.path.insert(0, str(repo_path))
            
            # Import EMAGE modules
            try:
                from models.motion_representation import MotionRep
                from models.audio_encoder import AudioEncoder
                from models.gesture_decoder import GestureDecoder
                from utils.config import Config
            except ImportError:
                # Fallback imports
                from EMAGE.models.motion_representation import MotionRep
                from EMAGE.models.audio_encoder import AudioEncoder
                from EMAGE.models.gesture_decoder import GestureDecoder
                from EMAGE.utils.config import Config
            
            # Load configuration
            config = Config(str(repo_path / "configs" / "emage_config.yaml"))
            
            # Initialize model components
            self.audio_encoder = AudioEncoder(
                input_dim=config.audio_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.latent_dim
            ).to(self.device)
            
            self.motion_decoder = GestureDecoder(
                latent_dim=config.latent_dim,
                motion_dim=config.motion_dim,
                hidden_dim=config.hidden_dim
            ).to(self.device)
            
            # Load pretrained weights
            checkpoint_path = repo_path / "checkpoints" / "emage_best.pth"
            if not checkpoint_path.exists():
                self._download_weights(checkpoint_path)
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.audio_encoder.load_state_dict(checkpoint['audio_encoder'])
            self.motion_decoder.load_state_dict(checkpoint['motion_decoder'])
            
            # Set to evaluation mode
            self.audio_encoder.eval()
            self.motion_decoder.eval()
            
            self.loaded = True
            logger.info("EMAGE model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load EMAGE model: {e}")
            raise
    
    def _download_weights(self, checkpoint_path: Path):
        """Download EMAGE pretrained weights"""
        from huggingface_hub import hf_hub_download
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download from Hugging Face
        downloaded_path = hf_hub_download(
            repo_id="PantoMatrix/EMAGE",
            filename="emage_best.pth",
            cache_dir=str(checkpoint_path.parent)
        )
        
        import shutil
        shutil.move(downloaded_path, str(checkpoint_path))
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Extract audio features for EMAGE"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=80,
            hop_length=320,  # 20ms hop
            win_length=800   # 50ms window
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize
        mel_normalized = (mel_db + 80) / 80
        mel_normalized = np.clip(mel_normalized, 0, 1)
        
        # Convert to tensor
        audio_features = torch.FloatTensor(mel_normalized).unsqueeze(0).to(self.device)
        
        return audio_features
    
    def generate_motion(
        self,
        audio_path: str,
        text: Optional[str] = None,
        emotion: str = "neutral",
        style: str = "natural",
        duration: Optional[float] = None
    ) -> np.ndarray:
        """Generate realistic body motion from audio"""
        if not self.loaded:
            self.load_model()
        
        # Preprocess audio
        audio_features = self.preprocess_audio(audio_path)
        
        # Encode audio to latent space
        with torch.no_grad():
            audio_latent = self.audio_encoder(audio_features)
            
            # Add emotion and style conditioning
            emotion_embedding = self._get_emotion_embedding(emotion)
            style_embedding = self._get_style_embedding(style)
            
            # Combine embeddings
            conditioned_latent = audio_latent + emotion_embedding + style_embedding
            
            # Decode to motion
            motion_sequence = self.motion_decoder(conditioned_latent)
            
            # Convert to numpy
            motion = motion_sequence.cpu().numpy()
        
        return motion
    
    def _get_emotion_embedding(self, emotion: str) -> torch.Tensor:
        """Get emotion embedding vector"""
        emotion_map = {
            "neutral": [0.0, 0.0, 0.0, 0.0],
            "happy": [1.0, 0.0, 0.0, 0.0],
            "sad": [0.0, 1.0, 0.0, 0.0],
            "angry": [0.0, 0.0, 1.0, 0.0],
            "excited": [0.0, 0.0, 0.0, 1.0]
        }
        
        emotion_vector = emotion_map.get(emotion, emotion_map["neutral"])
        return torch.tensor(emotion_vector, device=self.device).unsqueeze(0)
    
    def _get_style_embedding(self, style: str) -> torch.Tensor:
        """Get style embedding vector"""
        style_map = {
            "natural": [1.0, 0.0, 0.0],
            "formal": [0.0, 1.0, 0.0],
            "casual": [0.0, 0.0, 1.0]
        }
        
        style_vector = style_map.get(style, style_map["natural"])
        return torch.tensor(style_vector, device=self.device).unsqueeze(0)
    
    def motion_to_poses(self, motion: np.ndarray, fps: int = 30) -> np.ndarray:
        """Convert motion representation to 3D poses"""
        # Motion shape: (seq_len, motion_dim)
        # Convert to SMPL-X pose parameters
        
        seq_len = motion.shape[0]
        num_joints = 55  # SMPL-X joints
        
        # Reshape motion to joint rotations
        poses = motion.reshape(seq_len, num_joints, -1)
        
        # Convert to 3D coordinates if needed
        if poses.shape[-1] == 6:  # 6D rotation representation
            poses_3d = self._convert_6d_to_3d(poses)
        else:
            poses_3d = poses
        
        return poses_3d
    
    def _convert_6d_to_3d(self, poses_6d: np.ndarray) -> np.ndarray:
        """Convert 6D rotation to 3D coordinates"""
        seq_len, num_joints, _ = poses_6d.shape
        poses_3d = np.zeros((seq_len, num_joints, 3))
        
        # Simple conversion - in practice would use proper rotation matrices
        for i in range(seq_len):
            for j in range(num_joints):
                # Convert 6D rotation to 3D position (simplified)
                rot_6d = poses_6d[i, j]
                # Apply forward kinematics (simplified)
                poses_3d[i, j] = rot_6d[:3] * 0.1  # Scale factor
        
        return poses_3d
    
    def render_avatar(
        self,
        poses: np.ndarray,
        output_path: str,
        avatar_type: str = "realistic",
        background: Optional[str] = None
    ) -> str:
        """Render poses to realistic avatar video"""
        
        if avatar_type == "realistic":
            return self._render_smplx_avatar(poses, output_path, background)
        else:
            return self._render_simple_avatar(poses, output_path, background)
    
    def _render_smplx_avatar(
        self,
        poses: np.ndarray,
        output_path: str,
        background: Optional[str]
    ) -> str:
        """Render with SMPL-X realistic human model"""
        try:
            import smplx
            import pyrender
            import trimesh
            
            # Load SMPL-X model
            smplx_model = smplx.create(
                model_path="models/smplx",
                model_type="smplx",
                gender="neutral",
                use_face_contour=False
            )
            
            # Setup renderer
            renderer = pyrender.OffscreenRenderer(1280, 720)
            
            frames = []
            for i, pose in enumerate(poses):
                # Convert pose to SMPL-X parameters
                body_pose = torch.tensor(pose.reshape(-1)).unsqueeze(0).float()
                
                # Generate mesh
                output = smplx_model(body_pose=body_pose)
                vertices = output.vertices.detach().cpu().numpy()[0]
                faces = smplx_model.faces
                
                # Create mesh
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh.visual.face_colors = [200, 200, 250, 255]  # Skin color
                
                # Setup scene
                scene = pyrender.Scene()
                mesh_node = pyrender.Mesh.from_trimesh(mesh)
                scene.add(mesh_node)
                
                # Add lighting
                light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
                scene.add(light, pose=np.eye(4))
                
                # Add camera
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
                camera_pose = np.array([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.5],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                scene.add(camera, pose=camera_pose)
                
                # Render frame
                color, _ = renderer.render(scene)
                frames.append(color)
            
            # Write video
            self._write_video(frames, output_path)
            renderer.delete()
            
            return output_path
            
        except Exception as e:
            logger.warning(f"SMPL-X rendering failed: {e}, using simple rendering")
            return self._render_simple_avatar(poses, output_path, background)
    
    def _render_simple_avatar(
        self,
        poses: np.ndarray,
        output_path: str,
        background: Optional[str]
    ) -> str:
        """Simple stick figure rendering"""
        
        frames = []
        height, width = 720, 1280
        
        # Load background if provided
        bg_image = None
        if background and Path(background).exists():
            bg_image = cv2.imread(background)
            bg_image = cv2.resize(bg_image, (width, height))
        
        for pose in poses:
            # Create frame
            if bg_image is not None:
                frame = bg_image.copy()
            else:
                frame = np.ones((height, width, 3), dtype=np.uint8) * 240
            
            # Draw skeleton
            self._draw_skeleton(frame, pose, width, height)
            frames.append(frame)
        
        # Write video
        self._write_video(frames, output_path)
        return output_path
    
    def _draw_skeleton(self, frame: np.ndarray, pose: np.ndarray, width: int, height: int):
        """Draw 3D skeleton on frame"""
        
        # Project 3D to 2D
        pose_2d = pose[:, :2]  # Use X, Y coordinates
        
        # Normalize and scale
        pose_2d = (pose_2d + 2) / 4  # Normalize to [0, 1]
        pose_2d[:, 0] *= width
        pose_2d[:, 1] *= height
        
        # Define skeleton connections (simplified)
        connections = [
            # Spine
            (0, 1), (1, 2), (2, 3),
            # Arms
            (1, 4), (4, 5), (5, 6),  # Right arm
            (1, 7), (7, 8), (8, 9),  # Left arm
            # Legs
            (0, 10), (10, 11), (11, 12),  # Right leg
            (0, 13), (13, 14), (14, 15),  # Left leg
        ]
        
        # Draw connections
        for start, end in connections:
            if start < len(pose_2d) and end < len(pose_2d):
                pt1 = tuple(pose_2d[start].astype(int))
                pt2 = tuple(pose_2d[end].astype(int))
                cv2.line(frame, pt1, pt2, (0, 150, 255), 4)
        
        # Draw joints
        for i, point in enumerate(pose_2d):
            center = tuple(point.astype(int))
            color = (255, 0, 0) if i in [0, 1, 2, 3] else (0, 255, 0)  # Red for spine
            cv2.circle(frame, center, 6, color, -1)
    
    def _write_video(self, frames: List[np.ndarray], output_path: str, fps: int = 30):
        """Write frames to video file"""
        
        if not frames:
            raise ValueError("No frames to write")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    def generate_full_video(
        self,
        audio_path: str,
        output_path: str,
        text: Optional[str] = None,
        emotion: str = "neutral",
        style: str = "natural",
        avatar_type: str = "realistic",
        background: Optional[str] = None
    ) -> str:
        """Complete pipeline: audio -> motion -> rendered video"""
        
        logger.info("Generating realistic body expressions with EMAGE...")
        
        # Generate motion from audio
        motion = self.generate_motion(
            audio_path=audio_path,
            text=text,
            emotion=emotion,
            style=style
        )
        
        # Convert to poses
        poses = self.motion_to_poses(motion)
        
        # Render avatar
        video_path = self.render_avatar(
            poses=poses,
            output_path=output_path,
            avatar_type=avatar_type,
            background=background
        )
        
        # Add audio to video
        self._add_audio_to_video(video_path, audio_path)
        
        logger.info(f"EMAGE video generation completed: {video_path}")
        return video_path
    
    def _add_audio_to_video(self, video_path: str, audio_path: str):
        """Add audio track to video using ffmpeg"""
        try:
            import subprocess
            
            temp_video = video_path.replace('.mp4', '_temp.mp4')
            subprocess.run([
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                temp_video
            ], check=True, capture_output=True)
            
            # Replace original with audio version
            import os
            os.replace(temp_video, video_path)
            
        except Exception as e:
            logger.warning(f"Failed to add audio: {e}")
    
    def is_loaded(self) -> bool:
        return self.loaded and self.audio_encoder is not None

# Global instance
_emage_model = None

def get_emage_model() -> EMageRealistic:
    """Get or create global EMAGE model instance"""
    global _emage_model
    if _emage_model is None:
        _emage_model = EMageRealistic()
        _emage_model.load_model()
    return _emage_model