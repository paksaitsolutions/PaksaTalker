"""
Realistic Full Body Animation Models
Integration of state-of-the-art full body animation models
"""

import torch
import numpy as np
import cv2
from typing import Optional, Dict, Any, List, Tuple
import subprocess
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RealisticFullBodyAnimator:
    """Integration of multiple realistic full body animation models"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._get_device(device)
        self.models = {}
        self.loaded_models = []
        
    def _get_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_models(self):
        """Load available full body animation models in priority order"""
        
        # 1. EMAGE - Expressive Motion Avatar Generation (BEST)
        try:
            self._load_emage()
            self.loaded_models.append("EMAGE")
        except Exception as e:
            logger.warning(f"EMAGE loading failed: {e}")
        
        # 2. PantoMatrix - Full body gesture generation
        try:
            self._load_pantomatrix()
            self.loaded_models.append("PantoMatrix")
        except Exception as e:
            logger.warning(f"PantoMatrix loading failed: {e}")
        
        # 3. BEAT - Body Expression and Audio-driven Talking
        try:
            self._load_beat()
            self.loaded_models.append("BEAT")
        except Exception as e:
            logger.warning(f"BEAT loading failed: {e}")
        
        # 4. DiffMotion - Diffusion-based motion generation
        try:
            self._load_diffmotion()
            self.loaded_models.append("DiffMotion")
        except Exception as e:
            logger.warning(f"DiffMotion loading failed: {e}")
        
        logger.info(f"Loaded models: {self.loaded_models}")
    
    def _load_emage(self):
        """Load EMAGE - Expressive Motion Avatar Generation"""
        repo_path = Path("EMAGE")
        if not repo_path.exists():
            subprocess.run([
                "git", "clone", 
                "https://github.com/PantoMatrix/EMAGE.git"
            ], check=True)
        
        # Install dependencies
        subprocess.run([
            "pip", "install", "-r", 
            str(repo_path / "requirements.txt")
        ], check=True)
        
        # Load EMAGE model
        import sys
        sys.path.append(str(repo_path))
        
        from EMAGE.models.emage_model import EMageModel
        from EMAGE.utils.config import load_config
        
        config_path = repo_path / "configs" / "emage_config.yaml"
        config = load_config(str(config_path))
        
        self.models["EMAGE"] = EMageModel(config, device=self.device)
        self.models["EMAGE"].load_pretrained()
    
    def _load_pantomatrix(self):
        """Load PantoMatrix - Full body gesture generation"""
        repo_path = Path("PantoMatrix")
        if not repo_path.exists():
            subprocess.run([
                "git", "clone",
                "https://github.com/PantoMatrix/PantoMatrix.git"
            ], check=True)
        
        import sys
        sys.path.append(str(repo_path))
        
        from PantoMatrix.models.pantomatrix import PantoMatrixModel
        
        self.models["PantoMatrix"] = PantoMatrixModel(device=self.device)
        self.models["PantoMatrix"].load_model()
    
    def _load_beat(self):
        """Load BEAT - Body Expression and Audio-driven Talking"""
        repo_path = Path("BEAT")
        if not repo_path.exists():
            subprocess.run([
                "git", "clone",
                "https://github.com/PantoMatrix/BEAT.git"
            ], check=True)
        
        import sys
        sys.path.append(str(repo_path))
        
        from BEAT.models.beat_model import BEATModel
        
        self.models["BEAT"] = BEATModel(device=self.device)
        self.models["BEAT"].load_checkpoints()
    
    def _load_diffmotion(self):
        """Load DiffMotion - Diffusion-based motion generation"""
        repo_path = Path("DiffMotion")
        if not repo_path.exists():
            subprocess.run([
                "git", "clone",
                "https://github.com/mingyuan-zhang/MotionDiffuse.git",
                str(repo_path)
            ], check=True)
        
        import sys
        sys.path.append(str(repo_path))
        
        from MotionDiffuse.models.mdm import MDM
        
        self.models["DiffMotion"] = MDM(device=self.device)
        self.models["DiffMotion"].load_model()
    
    def generate_full_body_animation(
        self,
        audio_path: str,
        text: Optional[str] = None,
        style: str = "natural",
        duration: Optional[float] = None,
        fps: int = 30
    ) -> np.ndarray:
        """Generate realistic full body animation from audio and text"""
        
        # Try models in priority order
        for model_name in ["EMAGE", "PantoMatrix", "BEAT", "DiffMotion"]:
            if model_name in self.loaded_models:
                try:
                    return self._generate_with_model(
                        model_name, audio_path, text, style, duration, fps
                    )
                except Exception as e:
                    logger.warning(f"{model_name} generation failed: {e}")
                    continue
        
        # Fallback to basic animation
        return self._generate_basic_animation(audio_path, duration, fps)
    
    def _generate_with_model(
        self,
        model_name: str,
        audio_path: str,
        text: Optional[str],
        style: str,
        duration: Optional[float],
        fps: int
    ) -> np.ndarray:
        """Generate animation with specific model"""
        
        if model_name == "EMAGE":
            return self._generate_emage(audio_path, text, style, duration, fps)
        elif model_name == "PantoMatrix":
            return self._generate_pantomatrix(audio_path, text, style, duration, fps)
        elif model_name == "BEAT":
            return self._generate_beat(audio_path, text, style, duration, fps)
        elif model_name == "DiffMotion":
            return self._generate_diffmotion(audio_path, text, style, duration, fps)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _generate_emage(
        self,
        audio_path: str,
        text: Optional[str],
        style: str,
        duration: Optional[float],
        fps: int
    ) -> np.ndarray:
        """Generate with EMAGE model"""
        model = self.models["EMAGE"]
        
        # Preprocess audio
        audio_features = model.preprocess_audio(audio_path)
        
        # Generate motion
        motion_data = model.generate_motion(
            audio_features=audio_features,
            text=text,
            style=style,
            duration=duration
        )
        
        # Convert to pose sequence
        poses = model.motion_to_poses(motion_data, fps=fps)
        
        return poses
    
    def _generate_pantomatrix(
        self,
        audio_path: str,
        text: Optional[str],
        style: str,
        duration: Optional[float],
        fps: int
    ) -> np.ndarray:
        """Generate with PantoMatrix model"""
        model = self.models["PantoMatrix"]
        
        # Generate gestures
        gestures = model.generate_gestures(
            audio_path=audio_path,
            text=text,
            style=style,
            fps=fps
        )
        
        return gestures
    
    def _generate_beat(
        self,
        audio_path: str,
        text: Optional[str],
        style: str,
        duration: Optional[float],
        fps: int
    ) -> np.ndarray:
        """Generate with BEAT model"""
        model = self.models["BEAT"]
        
        # Generate body expressions
        expressions = model.generate_expressions(
            audio_path=audio_path,
            text=text,
            emotion=style,
            fps=fps
        )
        
        return expressions
    
    def _generate_diffmotion(
        self,
        audio_path: str,
        text: Optional[str],
        style: str,
        duration: Optional[float],
        fps: int
    ) -> np.ndarray:
        """Generate with DiffMotion model"""
        model = self.models["DiffMotion"]
        
        # Generate motion with diffusion
        motion = model.sample_motion(
            audio_path=audio_path,
            text_prompt=text,
            style=style,
            length=int(duration * fps) if duration else None
        )
        
        return motion
    
    def _generate_basic_animation(
        self,
        audio_path: str,
        duration: Optional[float],
        fps: int
    ) -> np.ndarray:
        """Fallback basic animation"""
        import librosa
        
        # Load audio to get duration
        if duration is None:
            audio, sr = librosa.load(audio_path)
            duration = len(audio) / sr
        
        num_frames = int(duration * fps)
        
        # Generate basic pose sequence (25 joints, 3D coordinates)
        poses = np.zeros((num_frames, 25, 3))
        
        # Add basic breathing and swaying motion
        for i in range(num_frames):
            t = i / fps
            
            # Breathing (chest movement)
            breathing = 0.02 * np.sin(t * 2 * np.pi * 0.3)  # 0.3 Hz
            poses[i, 1, 2] = breathing  # Chest Z movement
            
            # Swaying (hip movement)
            sway = 0.01 * np.sin(t * 2 * np.pi * 0.2)  # 0.2 Hz
            poses[i, 0, 0] = sway  # Hip X movement
            
            # Arm gestures (simple)
            arm_gesture = 0.05 * np.sin(t * 2 * np.pi * 0.5)  # 0.5 Hz
            poses[i, 3, 1] = arm_gesture  # Right arm Y
            poses[i, 6, 1] = -arm_gesture  # Left arm Y
        
        return poses
    
    def render_animation(
        self,
        poses: np.ndarray,
        output_path: str,
        background: Optional[np.ndarray] = None,
        avatar_model: str = "default"
    ) -> str:
        """Render pose sequence to video"""
        
        # Use SMPL-X or similar for realistic rendering
        try:
            return self._render_with_smplx(poses, output_path, background)
        except Exception as e:
            logger.warning(f"SMPL-X rendering failed: {e}")
            return self._render_basic(poses, output_path, background)
    
    def _render_with_smplx(
        self,
        poses: np.ndarray,
        output_path: str,
        background: Optional[np.ndarray]
    ) -> str:
        """Render with SMPL-X for realistic human model"""
        try:
            import smplx
            import trimesh
            
            # Load SMPL-X model
            smplx_model = smplx.create(
                model_path="models/smplx",
                model_type="smplx",
                gender="neutral",
                use_face_contour=False,
                use_pca=False
            )
            
            frames = []
            for pose in poses:
                # Convert pose to SMPL-X parameters
                body_pose = torch.tensor(pose.reshape(-1)).unsqueeze(0)
                
                # Generate mesh
                output = smplx_model(body_pose=body_pose)
                vertices = output.vertices.detach().cpu().numpy()[0]
                faces = smplx_model.faces
                
                # Render frame
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                frame = self._render_mesh_frame(mesh, background)
                frames.append(frame)
            
            # Write video
            self._write_video_frames(frames, output_path)
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"SMPL-X rendering failed: {e}")
    
    def _render_basic(
        self,
        poses: np.ndarray,
        output_path: str,
        background: Optional[np.ndarray]
    ) -> str:
        """Basic stick figure rendering"""
        
        frames = []
        height, width = 720, 1280
        
        for pose in poses:
            # Create frame
            if background is not None:
                frame = cv2.resize(background, (width, height))
            else:
                frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Draw stick figure
            self._draw_stick_figure(frame, pose, width, height)
            frames.append(frame)
        
        # Write video
        self._write_video_frames(frames, output_path)
        return output_path
    
    def _draw_stick_figure(self, frame: np.ndarray, pose: np.ndarray, width: int, height: int):
        """Draw stick figure on frame"""
        
        # Scale and center pose
        pose_2d = pose[:, :2]  # Use X, Y coordinates
        pose_2d = (pose_2d + 1) * 0.5  # Normalize to [0, 1]
        pose_2d[:, 0] *= width
        pose_2d[:, 1] *= height
        
        # Define skeleton connections
        connections = [
            (0, 1), (1, 2), (2, 3),  # Spine
            (1, 4), (4, 5), (5, 6),  # Right arm
            (1, 7), (7, 8), (8, 9),  # Left arm
            (0, 10), (10, 11), (11, 12),  # Right leg
            (0, 13), (13, 14), (14, 15),  # Left leg
        ]
        
        # Draw connections
        for start, end in connections:
            if start < len(pose_2d) and end < len(pose_2d):
                pt1 = tuple(pose_2d[start].astype(int))
                pt2 = tuple(pose_2d[end].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
        
        # Draw joints
        for point in pose_2d:
            center = tuple(point.astype(int))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    
    def _write_video_frames(self, frames: List[np.ndarray], output_path: str, fps: int = 30):
        """Write frames to video file"""
        
        if not frames:
            raise ValueError("No frames to write")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    def is_loaded(self) -> bool:
        """Check if any models are loaded"""
        return len(self.loaded_models) > 0

# Global instance
_full_body_animator = None

def get_full_body_animator() -> RealisticFullBodyAnimator:
    """Get or create global full body animator instance"""
    global _full_body_animator
    if _full_body_animator is None:
        _full_body_animator = RealisticFullBodyAnimator()
        _full_body_animator.load_models()
    return _full_body_animator