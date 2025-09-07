"""
Fusion Engine - Combines SadTalker face animation with optional EMAGE body animation.

This implementation:
- Resolves duplicate installs by locating model roots from known candidates
- Prefers local, already-installed models (no network calls)
- Improves fallback: if SadTalker fails, creates a base video then applies
  a lightweight Wav2Lip stub animation for visible lip movement
"""
import os
import sys
from pathlib import Path
from typing import Optional


class FusionEngine:
    def __init__(self):
        self.sadtalker_dir = self._locate_sadtalker()
        self.sadtalker_available = bool(self.sadtalker_dir)
        self.emage_available = self._check_emage()

    def _locate_sadtalker(self) -> Optional[Path]:
        """Find a local SadTalker directory with inference.py."""
        candidates = []
        env = os.getenv('PAKSA_SADTALKER_DIR')
        if env:
            candidates.append(Path(env))
        # Prefer top-level bundled repo
        candidates.append(Path('SadTalker'))
        # Some repos embed under PaksaTalker/
        candidates.append(Path('PaksaTalker') / 'SadTalker')
        seen = set()
        for p in candidates:
            if p in seen:
                continue
            seen.add(p)
            try:
                if p.exists() and (p / 'inference.py').exists():
                    return p
            except Exception:
                continue
        return None

    def _check_emage(self) -> bool:
        try:
            from models.emage_realistic import emage_available
            return emage_available()
        except Exception:
            return False

    def _ffprobe_duration(self, audio_path: str, default: int = 10) -> int:
        import subprocess
        try:
            result = subprocess.run([
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0", audio_path
            ], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return max(3, int(float(result.stdout.strip())))
        except Exception:
            pass
        return default

    def _make_base_video(self, face_image: str, audio_path: str, output_path: str, fps: int) -> str:
        """Create a simple talking-head base video (still image + audio)."""
        import subprocess
        duration = self._ffprobe_duration(audio_path)
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", face_image,
            "-i", audio_path,
            "-c:v", "libx264", "-t", str(duration), "-pix_fmt", "yuv420p",
            "-r", str(fps), "-shortest", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:2000]}")
        return output_path

    def _apply_wav2lip_stub(self, input_video: str, audio_path: str, output_path: str, fps: int) -> Optional[str]:
        """Apply the lightweight local Wav2Lip stub to animate lips.

        Returns output path on success, or None to keep the input video.
        """
        try:
            from src.wav2lip.inference import Wav2Lip
        except Exception:
            return None
        try:
            os.makedirs(Path(output_path).parent, exist_ok=True)
            w2l = Wav2Lip(checkpoint_path=str(Path('models') / 'wav2lip' / 'wav2lip.pth'))
            res = w2l.inference(face=input_video, audio=audio_path, outfile=output_path, fps=fps)
            if res and Path(res).exists():
                return res
        except Exception:
            return None
        return None

    def generate(
        self,
        face_image: str,
        audio_path: str,
        output_path: str,
        emotion: str = "neutral",
        style: str = "natural",
        fps: int = 25,
        resolution: str = "720p",
        prefer_wav2lip2: bool = False,
        use_emage: Optional[bool] = None,
        preprocess: Optional[str] = None
    ) -> str:
        """Generate fusion video combining face and optional body animation."""

        print("[Fusion] Start")
        print(f"  image={face_image}")
        print(f"  audio={audio_path}")
        print(f"  fps={fps} res={resolution} emotion={emotion} style={style}")
        print(f"  prefer_wav2lip2={prefer_wav2lip2} use_emage={use_emage}")
        if preprocess:
            print(f"  preprocess={preprocess}")

        face_video = None
        # 1) Try SadTalker CLI if available locally
        if self.sadtalker_available:
            try:
                from real_video_generator import generate_real_video
                temp_face = str(Path(output_path).with_name(Path(output_path).stem + '_face.mp4'))
                print("[Fusion] SadTalker running...")
                # Ensure real_video_generator respects local SadTalker dir via env
                if self.sadtalker_dir:
                    os.environ.setdefault('PAKSA_SADTALKER_DIR', str(self.sadtalker_dir))
                face_video = generate_real_video(face_image, audio_path, temp_face, preprocess=preprocess)
                print(f"[Fusion] SadTalker OK: {face_video}")
            except Exception as e:
                print(f"[Fusion] SadTalker failed: {e}")

        # 2) Optional EMAGE body video (placeholder availability)
        body_video = None
        if use_emage and self.emage_available:
            try:
                print("[Fusion] EMAGE requested...")
                from models.emage_realistic import EMageRealistic
                emage = EMageRealistic()
                temp_body = str(Path(output_path).with_name(Path(output_path).stem + '_body.mp4'))
                body_video = emage.generate_full_video(
                    audio_path=audio_path,
                    emotion=emotion,
                    style=style,
                    output_path=temp_body
                )
                print(f"[Fusion] EMAGE OK: {body_video}")
            except Exception as e:
                print(f"[Fusion] EMAGE failed: {e}")

        # 3) If no face video yet, create base video and attempt Wav2Lip stub
        if not face_video:
            print("[Fusion] Building base video via ffmpeg...")
            base = self._make_base_video(face_image, audio_path, output_path, fps)
            # Try to animate lips slightly using local stub (no heavy deps)
            try:
                temp_out = str(Path(output_path).with_name(Path(output_path).stem + '_w2l.mp4'))
                w2l = self._apply_wav2lip_stub(base, audio_path, temp_out, fps)
                if w2l:
                    # Move to final output
                    import shutil
                    shutil.copy2(w2l, output_path)
                    print(f"[Fusion] Wav2Lip stub applied: {output_path}")
                else:
                    print("[Fusion] Wav2Lip stub unavailable; keeping base video")
                return output_path
            except Exception as e:
                raise Exception(f"Video generation failed: {e}")

        # 4) Composite face + body if both available; else use face only
        if body_video and face_video:
            print("[Fusion] Compositing face + body...")
            try:
                import subprocess
                cmd = [
                    "ffmpeg", "-y",
                    "-i", face_video,
                    "-i", body_video,
                    "-filter_complex", "[0:v][1:v]overlay=0:0",
                    "-c:v", "libx264", "-c:a", "aac",
                    output_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr[:2000])
                print(f"[Fusion] Composite OK: {output_path}")
            except Exception as e:
                print(f"[Fusion] Composite failed, using face only: {e}")
                import shutil
                shutil.copy2(face_video, output_path)
        else:
            import shutil
            shutil.copy2(face_video, output_path)
            print(f"[Fusion] Face-only output: {output_path}")

        return output_path
