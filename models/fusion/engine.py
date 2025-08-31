import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import shutil
import logging


def _safe_mkdir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _read_video_frames(path: str) -> Tuple[cv2.VideoCapture, int, int, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    return cap, width, height, fps


def _find_ffmpeg() -> Optional[str]:
    try:
        import subprocess
        candidates = [
            'ffmpeg',
            r"C:\\ffmpeg\\bin\\ffmpeg.exe",
            r"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
            os.path.expandvars(r"C:\\Users\\%USERNAME%\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.0-full_build\\bin\\ffmpeg.exe"),
        ]
        for p in candidates:
            try:
                res = subprocess.run([p, '-version'], capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    return p
            except Exception:
                continue
    except Exception:
        pass
    return None


def _extract_audio(video_path: str, audio_out: str) -> bool:
    import subprocess
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        logging.getLogger(__name__).warning("ffmpeg not found; cannot extract audio")
        return False
    _safe_mkdir(Path(audio_out))
    cmd = [
        ffmpeg, '-y',
        '-i', video_path,
        '-vn', '-acodec', 'copy',
        audio_out
    ]
    try:
        res = subprocess.run(cmd, capture_output=True)
        return res.returncode == 0 and Path(audio_out).exists()
    except Exception as e:
        logging.getLogger(__name__).warning(f"Audio extraction failed: {e}")
        return False


def _merge_audio(video_in: str, audio_in: str, video_out: str) -> str:
    import subprocess
    ffmpeg = _find_ffmpeg()
    _safe_mkdir(Path(video_out))
    if not ffmpeg:
        # Fallback: copy silent video if ffmpeg not found
        try:
            shutil.copyfile(video_in, video_out)
            logging.getLogger(__name__).warning("ffmpeg not found; returning silent video")
            return video_out
        except Exception as e:
            raise RuntimeError(f"Failed to provide output without ffmpeg: {e}")
    cmd = [
        ffmpeg, '-y',
        '-i', video_in,
        '-i', audio_in,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        video_out
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        # Fallback to silent
        shutil.copyfile(video_in, video_out)
        logging.getLogger(__name__).warning("ffmpeg merge failed; returning silent video")
    return video_out


def _oval_mask(w: int, h: int, feather: int = 25) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    axes = (int(w * 0.45), int(h * 0.6))
    center = (w // 2, h // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather | 1, feather | 1), 0)
    return mask


def _color_transfer_lab(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Match color statistics of src to target in LAB space (Reinhard-style)."""
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute channel-wise mean/std
    s_mean, s_std = cv2.meanStdDev(src_lab)
    t_mean, t_std = cv2.meanStdDev(tgt_lab)
    s_mean, s_std = s_mean.flatten(), (s_std.flatten() + 1e-6)
    t_mean, t_std = t_mean.flatten(), (t_std.flatten() + 1e-6)

    result = src_lab.copy()
    for c in range(3):
        result[..., c] = (result[..., c] - s_mean[c]) * (t_std[c] / s_std[c]) + t_mean[c]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def _bbox_from_landmarks(lms: np.ndarray, pad: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = max(0, int(np.min(lms[:, 0]) - pad))
    y1 = max(0, int(np.min(lms[:, 1]) - pad))
    x2 = min(W - 1, int(np.max(lms[:, 0]) + pad))
    y2 = min(H - 1, int(np.max(lms[:, 1]) + pad))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _track_head_bboxes(video_path: str) -> List[Tuple[int, int, int, int]]:
    """Return per-frame head bbox using OpenSeeFace if available; otherwise a centered box."""
    cap, W, H, _ = _read_video_frames(video_path)
    bboxes: List[Tuple[int, int, int, int]] = []
    try:
        from OpenSeeFace.tracker import Tracker  # type: ignore
        # Try to ensure OSF models exist and resolve model_dir
        model_dir = None
        try:
            from models.emotion.model_loader import ensure_openseeface_models  # type: ignore
            model_dir = ensure_openseeface_models()
        except Exception:
            model_dir = None
        if not model_dir:
            import os as _os
            env_root = _os.environ.get('PAKSA_OSF_ROOT') or _os.environ.get('OPENSEEFACE_ROOT')
            model_dir = env_root if env_root else str(Path('OpenSeeFace') / 'models')
        tr = Tracker(W, H, max_faces=1, silent=True, model_dir=model_dir)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = tr.predict(frame)
            if faces:
                lms = np.array([[p[0], p[1]] for p in faces[0].lms], dtype=np.float32)
                x, y, w, h = _bbox_from_landmarks(lms, pad=12, W=W, H=H)
            else:
                # fallback center box
                w = int(W * 0.28)
                h = int(H * 0.28)
                x = W // 2 - w // 2
                y = int(H * 0.18)
            bboxes.append((x, y, w, h))
    except Exception:
        # No OpenSeeFace; simple static upper-center box
        w = int(W * 0.28)
        h = int(H * 0.28)
        x = W // 2 - w // 2
        y = int(H * 0.18)
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            bboxes.append((x, y, w, h))
    cap.release()
    return bboxes


def _parse_resolution(resolution: str) -> Tuple[int, int]:
    res_map = {
        '360p': (640, 360),
        '480p': (854, 480),
        '720p': (1280, 720),
        '1080p': (1920, 1080),
    }
    if isinstance(resolution, str) and resolution.lower() in res_map:
        return res_map[resolution.lower()]
    # Try "WxH"
    if isinstance(resolution, str) and 'x' in resolution:
        try:
            w, h = resolution.lower().split('x')
            return int(w), int(h)
        except Exception:
            pass
    return (1280, 720)


def _placeholder_body_video(out_path: str, duration_sec: float, fps: float, size: Tuple[int, int]) -> str:
    W, H = size
    total = int(max(1, round(duration_sec * fps)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    _safe_mkdir(Path(out_path))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for i in range(total):
        t = i / fps
        # Simple animated gradient background
        base = np.linspace(0, 255, W, dtype=np.uint8)
        frame = np.tile(base, (H, 1))
        frame = np.stack([np.roll(frame, int(30*np.sin(t*2)), axis=1), frame, np.roll(frame, int(50*np.cos(t*1.3)), axis=0)], axis=2)
        # Add a subtle ellipse for torso area
        overlay = frame.copy()
        center = (W//2, int(H*0.55))
        axes = (int(W*0.18), int(H*0.28))
        color = (200, 200, 210)
        cv2.ellipse(overlay, center, axes, 0, 0, 360, color, -1)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
        writer.write(frame.astype(np.uint8))
    writer.release()
    return out_path


def _loop_face_image(image_path: str, out_path: str, duration_sec: float, fps: float, size: Tuple[int, int]):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Cannot read face image")
    H, W = size[1], size[0]
    frame = cv2.resize(img, (W, H))
    total = int(max(1, round(duration_sec * fps)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for _ in range(total):
        writer.write(frame)
    writer.release()


class FusionEngine:
    """Compose SadTalker face over EMAGE body using head tracking."""

    def __init__(self):
        pass

    def generate(
        self,
        face_image: str,
        audio_path: str,
        output_path: str,
        emotion: str = 'neutral',
        style: str = 'natural',
        fps: int = 25,
        resolution: str = '720p',
        prefer_wav2lip2: bool = False,
        use_emage: Optional[bool] = None
    ) -> str:
        log = logging.getLogger(__name__)

        # 0) Setup
        temp_dir = Path('temp')
        temp_dir.mkdir(exist_ok=True)
        body_path = str(temp_dir / 'fusion_body.mp4')

        # 1) Generate face track first (more robust; has audio duration)
        face_track = str(temp_dir / 'fusion_face.mp4')
        generated = False
        if prefer_wav2lip2:
            try:
                from models.wav2lip2_aoti import get_wav2lip2_model
                w2l = get_wav2lip2_model()
                face_track = w2l.generate_video(
                    image_path=face_image,
                    audio_path=audio_path,
                    output_path=face_track,
                    fps=int(round(fps))
                )
                generated = True
            except Exception:
                generated = False
        if not generated:
            try:
                from models.sadtalker_full import SadTalkerFull
                st = SadTalkerFull()
                face_track = st.generate(
                    image_path=face_image,
                    audio_path=audio_path,
                    output_path=face_track,
                    emotion=emotion,
                    enhance_face=True
                )
                generated = True
            except Exception:
                generated = False
        if not generated:
            # Loop still image to match duration
            # Try estimate duration from audio; otherwise default to 5s
            try:
                duration = 5.0
                try:
                    import librosa  # type: ignore
                    duration = float(librosa.get_duration(path=audio_path))
                except Exception:
                    pass
                fps_use = float(fps)
                # Use 720p as default until body is created
                Wp, Hp = _parse_resolution(resolution)
                _loop_face_image(face_image, face_track, duration, fps_use, (Wp, Hp))
            except Exception as e:
                raise RuntimeError(f"Failed to prepare face track: {e}")

        # Read face track properties
        try:
            cap_f_probe, WF_probe, HF_probe, face_fps = _read_video_frames(face_track)
            cap_f_probe.release()
        except Exception:
            WF_probe, HF_probe, face_fps = _parse_resolution(resolution)[0], _parse_resolution(resolution)[1], float(fps)

        fps_use = face_fps or float(fps)

        # 2) Generate body track (EMAGE preferred, unless disabled/unavailable)
        body_video = None
        # Determine whether to use EMAGE
        emage_allowed = True
        if use_emage is not None:
            emage_allowed = bool(use_emage)
        env_disable = os.environ.get('PAKSA_DISABLE_EMAGE') in ('1', 'true', 'True', 'yes')
        if env_disable:
            emage_allowed = False
        # If allowed, ensure it appears available
        if emage_allowed:
            try:
                from models.emage_realistic import emage_available  # type: ignore
                if not emage_available():
                    emage_allowed = False
            except Exception:
                emage_allowed = False

        if emage_allowed:
            try:
                from models.emage_realistic import get_emage_model
                emage = get_emage_model()
                body_video = emage.generate_full_video(
                    audio_path=audio_path,
                    output_path=body_path,
                    emotion=emotion,
                    style=style,
                    avatar_type='realistic'
                )
            except Exception as e:
                log.warning(f"EMAGE body generation failed; using placeholder. Reason: {e}")
                body_video = None

        if body_video is None:
            # Placeholder body track
            try:
                cap_tmp, Wp, Hp, _ = _read_video_frames(face_track)
                total = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                cap_tmp.release()
                if total <= 0:
                    raise RuntimeError("Cannot determine face track length")
                duration = total / fps_use
            except Exception:
                duration = 5.0
                Wp, Hp = _parse_resolution(resolution)
            log.info("Using placeholder body track (EMAGE disabled or unavailable)")
            body_video = _placeholder_body_video(body_path, duration, fps_use, (Wp or WF_probe, Hp or HF_probe))

        # 3) Head bboxes per frame
        bboxes = _track_head_bboxes(body_video)
        cap_b, W, H, body_fps = _read_video_frames(body_video)
        fps_use = body_fps or fps_use

        # 4) Composite per frame using bbox and oval mask
        cap_f, WF, HF, _ = _read_video_frames(face_track)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        tmp_video = str(temp_dir / 'fusion_composite_silent.mp4')
        _safe_mkdir(Path(tmp_video))
        writer = cv2.VideoWriter(tmp_video, fourcc, fps_use, (W, H))

        idx = 0
        while True:
            ret_b, frame_b = cap_b.read()
            ret_f, frame_f = cap_f.read()
            if not ret_b or not ret_f:
                break
            x, y, w, h = bboxes[idx] if idx < len(bboxes) else (W//2 - 100, int(H*0.2), 200, 200)
            idx += 1

            # Resize face and create mask
            face_resized = cv2.resize(frame_f, (w, h))
            # Color match face to ROI for better tone/lighting
            roi = frame_b[y:y+h, x:x+w]
            try:
                face_resized = _color_transfer_lab(face_resized, roi)
            except Exception:
                pass
            mask = _oval_mask(w, h, feather=31)
            mask_f = (mask.astype(np.float32) / 255.0)[..., None]

            # ROI blend
            if roi.shape[0] <= 0 or roi.shape[1] <= 0:
                writer.write(frame_b)
                continue
            blend = (mask_f * face_resized.astype(np.float32) + (1.0 - mask_f) * roi.astype(np.float32)).astype(np.uint8)
            frame_b[y:y+h, x:x+w] = blend
            writer.write(frame_b)

        cap_b.release()
        cap_f.release()
        writer.release()

        # 5) Mux audio back (prefer original input audio for reliability)
        final_out = output_path
        try:
            _merge_audio(tmp_video, audio_path, final_out)
        except Exception as e:
            log.warning(f"Merging audio failed, returning silent video: {e}")
            shutil.copyfile(tmp_video, final_out)
        return final_out
