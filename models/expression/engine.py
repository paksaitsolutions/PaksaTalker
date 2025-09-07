"""
Expression detection engines for the UI: MediaPipe (blendshapes), 3DDFA (3DMM),
OpenSeeFace (landmarks), and mini-XCEPTION (emotions).

The implementations are lightweight and defensive: if a real dependency is not
available, we return neutral/stub values rather than raising.
"""
from pathlib import Path
from typing import Dict, Any, Tuple


class ExpressionResult:
    def __init__(self, engine: str, blendshapes: Dict[str, float] | None = None, emotions: Dict[str, float] | None = None):
        self.engine = engine
        self.blendshapes = blendshapes or {}
        self.emotions = emotions or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blendshapes": self.blendshapes,
            "emotion_probs": self.emotions,
        }


def _normalize_engine(engine: str) -> str:
    e = (engine or "").strip().lower()
    if e in ("auto", "default", "best"):
        return "auto"
    if e in ("mediapipe", "mp", "blendshape", "blendshapes"):
        return "mediapipe"
    if e in ("3ddfa", "3dmm", "threeddfa", "3ddfa_v2"):
        return "threeddfa"
    if e in ("openseeface", "osf", "landmarks"):
        return "openseeface"
    if e in ("mini-xception", "mini_xception", "xception", "fer"):
        return "mini_xception"
    return e


def _stub_blendshapes() -> Dict[str, float]:
    return {"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.1, "mouthSmile": 0.3, "browInnerUp": 0.2}


def _stub_emotions() -> Dict[str, float]:
    return {"neutral": 0.7, "happy": 0.2, "sad": 0.1}


def detect_capabilities() -> Dict[str, bool]:
    """Detect available expression engines with best-effort checks."""
    caps: Dict[str, bool] = {}

    # MediaPipe (blendshapes via face mesh heuristics)
    try:
        import mediapipe  # noqa: F401
        caps["mediapipe"] = True
    except Exception:
        caps["mediapipe"] = False

    # OpenSeeFace (require tracker and at least one ONNX model)
    osf_root = Path("OpenSeeFace")
    osf_models = []
    try:
        osf_models = list((osf_root / "models").glob("*.onnx")) if osf_root.exists() else []
    except Exception:
        osf_models = []
    caps["openseeface"] = osf_root.exists() and (osf_root / "tracker.py").exists() and len(osf_models) > 0

    # 3DDFA V2 (repo + torch + checkpoints)
    try:
        import torch  # noqa: F401
        repo = Path("3DDFA_V2")
        caps["threeddfa"] = (
            repo.exists()
            and (repo / "TDDFA.py").exists()
            and (repo / "checkpoints" / "phase1_wpdc_vdc.pth.tar").exists()
            and (repo / "configs" / "bfm_noneck_v3.pkl").exists()
        )
    except Exception:
        caps["threeddfa"] = False

    # mini-XCEPTION (FER stub available)
    caps["mini_xception"] = (Path("models") / "emotion" / "fer_model.py").exists()

    return caps


def _estimate_emotions(image_path: str) -> Dict[str, float]:
    # Use local FER stub if available
    try:
        from models.emotion.fer_model import predict_emotions_from_image
        return predict_emotions_from_image(image_path)
    except Exception:
        return _stub_emotions()


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    import math
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _estimate_mediapipe(image_path: str) -> ExpressionResult:
    """Estimate a few blendshape-like values with MediaPipe Face Mesh landmarks.

    This is a heuristic approximation intended for UI feedback; values are
    scaled to 0..1 ranges but not calibrated.
    """
    try:
        import cv2
        import mediapipe as mp
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Could not read image")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        res = mesh.process(rgb)
        mesh.close()
        if not res.multi_face_landmarks:
            return ExpressionResult("mediapipe", _stub_blendshapes(), _stub_emotions())
        lm = res.multi_face_landmarks[0].landmark

        # Landmark helper to pixel coords
        def P(i: int) -> Tuple[float, float]:
            return (lm[i].x * w, lm[i].y * h)

        # Eyes: EAR-based blink (indices from MediaPipe Face Mesh)
        L = {"l": (33, 133, 159, 145), "r": (362, 263, 386, 374)}
        ear_vals = []
        for key in ("l", "r"):
            left, right, top, bottom = L[key]
            horiz = _euclidean(P(left), P(right)) + 1e-6
            vert = _euclidean(P(top), P(bottom))
            ear = (vert / horiz)
            # Map open~0.28, closed~0.12 to blink probability 0..1
            blink = max(0.0, min(1.0, (0.25 - ear) / 0.15))
            ear_vals.append((key, blink))
        eyeBlinkLeft = next(v for k, v in ear_vals if k == "l")
        eyeBlinkRight = next(v for k, v in ear_vals if k == "r")

        # Mouth smile proxy: width/height ratio
        # Corners 61 (left), 291 (right); lips 13 (upper), 14 (lower)
        width = _euclidean(P(61), P(291))
        height = _euclidean(P(13), P(14)) + 1e-6
        ratio = width / height
        # Map ratio ~ (1.5..4) into 0..1
        mouthSmile = max(0.0, min(1.0, (ratio - 1.8) / 2.2))

        # Brow inner up proxy: inner brow vs eye distance normalized by face size
        # Inner brows 66 (left), 296 (right); eye centers approx 159/386
        brow_l = P(66)
        brow_r = P(296)
        eye_l = ((P(159)[0] + P(145)[0]) / 2, (P(159)[1] + P(145)[1]) / 2)
        eye_r = ((P(386)[0] + P(374)[0]) / 2, (P(386)[1] + P(374)[1]) / 2)
        face_h = _euclidean(P(10), P(152)) + 1e-6  # forehead to chin
        brow_eye = (_euclidean(brow_l, eye_l) + _euclidean(brow_r, eye_r)) / 2.0
        # Smaller distance => brows down; larger => up. Normalize roughly.
        browInnerUp = max(0.0, min(1.0, (0.18 * face_h - brow_eye) / (0.08 * face_h)))

        blend = {
            "eyeBlinkLeft": float(eyeBlinkLeft),
            "eyeBlinkRight": float(eyeBlinkRight),
            "mouthSmile": float(mouthSmile),
            "browInnerUp": float(browInnerUp),
        }
        return ExpressionResult("mediapipe", blend, _estimate_emotions(image_path))
    except Exception:
        return ExpressionResult("mediapipe", _stub_blendshapes(), _estimate_emotions(image_path))


def _estimate_threeddfa(image_path: str) -> ExpressionResult:
    """Best-effort 3DDFA estimation. Falls back to stubs on any error."""
    try:
        # Minimal import to avoid heavy global side-effects
        import cv2
        from pathlib import Path as _P
        if not (_P("3DDFA_V2") / "TDDFA.py").exists():
            raise RuntimeError("3DDFA_V2 not present")
        # Lazy import inside try
        import sys as _sys
        _sys.path.insert(0, str(_P("3DDFA_V2").resolve()))
        from TDDFA import TDDFA
        from FaceBoxes import FaceBoxes

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Could not read image")
        face_boxes = FaceBoxes()
        boxes = face_boxes(img)
        if len(boxes) == 0:
            return ExpressionResult("threeddfa", _stub_blendshapes(), _estimate_emotions(image_path))
        tddfa = TDDFA()
        param_lst, roi_box_lst = tddfa(img, boxes)
        # Basic coefficients extraction
        if not param_lst:
            return ExpressionResult("threeddfa", _stub_blendshapes(), _estimate_emotions(image_path))
        # 3DDFA params contain exp and pose in different formats; for UI we only map a couple proxies.
        # Use mouth openness proxy from distance again (keeps consistent UI experience)
        h, w = img.shape[:2]
        # Reuse mediapipe heuristic for stable UI blendshapes, even if 3DDFA is available
        # (otherwise users see zeros due to coefficient mapping differences)
        return _estimate_mediapipe(image_path)
    except Exception:
        return ExpressionResult("threeddfa", _stub_blendshapes(), _estimate_emotions(image_path))


def _estimate_openseeface(image_path: str) -> ExpressionResult:
    # For now, provide landmark-based stub using MediaPipe to keep things simple
    # unless a full OpenSeeFace runtime is present.
    osf_root = Path("OpenSeeFace")
    if osf_root.exists():
        try:
            return _estimate_mediapipe(image_path)
        except Exception:
            pass
    return ExpressionResult("openseeface", _stub_blendshapes(), _estimate_emotions(image_path))


def _estimate_mini_xception(image_path: str) -> ExpressionResult:
    return ExpressionResult("mini_xception", {}, _estimate_emotions(image_path))


def estimate_from_path(image_path: str, engine: str = "auto") -> ExpressionResult:
    """Estimate expressions from an image path using the selected engine.

    Engines: "mediapipe", "threeddfa" (aka 3ddfa), "openseeface", "mini_xception", or "auto".
    """
    eng = _normalize_engine(engine)
    caps = detect_capabilities()

    if eng == "auto":
        # Prefer mediapipe for responsiveness, then 3DDFA, OSF, then emotions-only.
        eng = "mediapipe" if caps.get("mediapipe") else ("threeddfa" if caps.get("threeddfa") else ("openseeface" if caps.get("openseeface") else "mini_xception"))

    if eng == "mediapipe":
        return _estimate_mediapipe(image_path)
    if eng == "threeddfa":
        return _estimate_threeddfa(image_path)
    if eng == "openseeface":
        return _estimate_openseeface(image_path)
    if eng == "mini_xception":
        return _estimate_mini_xception(image_path)

    # Unknown engine: fall back to mediapipe or stubs
    return _estimate_mediapipe(image_path) if caps.get("mediapipe") else ExpressionResult("auto", _stub_blendshapes(), _stub_emotions())
