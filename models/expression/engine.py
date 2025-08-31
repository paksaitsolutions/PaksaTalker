import importlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Tuple


@dataclass
class ExpressionResult:
    engine: str
    blendshapes: Optional[Dict[str, float]] = None
    landmarks2d: Optional[List[Tuple[float, float]]] = None
    head_pose: Optional[Dict[str, float]] = None
    expression_params: Optional[Dict[str, float]] = None
    emotion_probs: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _has(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def _openseeface_available() -> bool:
    # Try importing OSF modules explicitly
    for m in ("OpenSeeFace.tracker", "OpenSeeFace.facetracker"):
        if _has(m):
            return True
    return False


def detect_capabilities() -> Dict[str, bool]:
    return {
        "mediapipe": _has("mediapipe"),
        "openseeface": _openseeface_available(),
        "threeddfa": _has("TDDFA_V2") or _has("3DDFA_V2") or _has("TDDFA"),
        "mini_xception": _has("models.emotion.fer_model") or _has("keras") or _has("onnxruntime"),
    }


def estimate_from_path(image_path: str, engine: str = "auto") -> ExpressionResult:
    caps = detect_capabilities()
    chosen = engine
    if engine == "auto":
        if caps.get("mediapipe"):
            chosen = "mediapipe"
        elif caps.get("threeddfa"):
            chosen = "threeddfa"
        elif caps.get("openseeface"):
            chosen = "openseeface"
        else:
            chosen = "mini_xception" if caps.get("mini_xception") else "none"

    # Dispatch
    if chosen == "mediapipe" and caps.get("mediapipe"):
        return _estimate_mediapipe(image_path)
    if chosen == "threeddfa" and caps.get("threeddfa"):
        return _estimate_3ddfa(image_path)
    if chosen == "openseeface" and caps.get("openseeface"):
        return _estimate_openseeface(image_path)
    if chosen == "mini_xception" and caps.get("mini_xception"):
        return _estimate_minixception(image_path)

    return ExpressionResult(engine=chosen)


def _estimate_mediapipe(image_path: str) -> ExpressionResult:
    try:
        import mediapipe as mp  # type: ignore
        import cv2  # type: ignore
        img = cv2.imread(image_path)
        if img is None:
            return ExpressionResult(engine="mediapipe")

        mp_face = mp.tasks.vision
        base_options = mp.tasks.BaseOptions(model_asset_path=None)
        options = mp_face.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        landmarker = mp_face.FaceLandmarker.create_from_options(options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)

        blend = {}
        if result.face_blendshapes:
            for cat in result.face_blendshapes[0].categories:
                blend[cat.category_name] = float(cat.score)

        lms = []
        if result.face_landmarks:
            for lm in result.face_landmarks[0]:
                lms.append((float(lm.x), float(lm.y)))

        # Head pose placeholder (requires solvePnP if needed)
        head = None
        return ExpressionResult(engine="mediapipe", blendshapes=blend or None, landmarks2d=lms or None, head_pose=head)
    except Exception:
        return ExpressionResult(engine="mediapipe")


def _estimate_3ddfa(image_path: str) -> ExpressionResult:
    # Lightweight wrapper; actual 3DDFA integration requires model init and inference
    try:
        # defer heavy imports
        from pathlib import Path
        if not Path(image_path).exists():
            return ExpressionResult(engine="threeddfa")
        # Placeholder: return empty but marked engine
        return ExpressionResult(engine="threeddfa", expression_params={})
    except Exception:
        return ExpressionResult(engine="threeddfa")


def _estimate_openseeface(image_path: str) -> ExpressionResult:
    try:
        import os
        import cv2
        from OpenSeeFace.tracker import Tracker  # type: ignore
        try:
            # Try to ensure models are present and discover model_dir
            from models.emotion.model_loader import ensure_openseeface_models  # type: ignore
            md = ensure_openseeface_models()
        except Exception:
            md = None

        frame = cv2.imread(image_path)
        if frame is None:
            return ExpressionResult(engine="openseeface")
        h, w = frame.shape[:2]

        # Prefer env override or ensured models dir
        model_dir = md or os.environ.get('PAKSA_OSF_ROOT') or os.environ.get('OPENSEEFACE_ROOT')
        if not model_dir:
            model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'OpenSeeFace', 'models')
            model_dir = os.path.abspath(model_dir)

        # Create tracker (single image; set small thread count and silent)
        tr = Tracker(w, h, max_faces=1, silent=True, model_dir=model_dir)
        faces = tr.predict(frame)
        if not faces:
            return ExpressionResult(engine="openseeface")

        face = faces[0]
        # landmarks are Nx3; take x,y
        try:
            lms = [(float(x), float(y)) for (x, y, *_ ) in face.lms]
        except Exception:
            # format may be different; try alternative
            l_arr = []
            for pt in face.lms:
                try:
                    l_arr.append((float(pt[0]), float(pt[1])))
                except Exception:
                    pass
            lms = l_arr

        head = None
        try:
            # face.euler is (pitch, yaw, roll) in degrees typically
            pitch, yaw, roll = face.euler
            head = {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}
        except Exception:
            head = None

        return ExpressionResult(engine="openseeface", landmarks2d=lms or None, head_pose=head)
    except Exception:
        return ExpressionResult(engine="openseeface")


def _estimate_minixception(image_path: str) -> ExpressionResult:
    # Try user-provided emotion detection function; otherwise return engine tag only
    try:
        # Prefer a pluggable function if available
        from models.emotion.fer_model import predict_emotions_from_image  # type: ignore
        probs = predict_emotions_from_image(image_path)
        return ExpressionResult(engine="mini_xception", emotion_probs=probs)
    except Exception:
        try:
            # If a loader is present but no model code, just report engine
            from models.emotion import model_loader  # type: ignore
            _ = model_loader.get_model_path()
            return ExpressionResult(engine="mini_xception")
        except Exception:
            return ExpressionResult(engine="mini_xception")
