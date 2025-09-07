"""
EMAGE Realistic - Minimal availability check and placeholder integration.

This module avoids network access and selects from multiple possible EMAGE
install locations to reduce duplication conflicts.
"""
import os
from pathlib import Path
from typing import Optional


def _candidate_emage_paths() -> list[Path]:
    """Return plausible local EMAGE repo locations in priority order."""
    env = os.getenv('PAKSA_EMAGE_ROOT')
    candidates = []
    if env:
        candidates.append(Path(env))
    # Prefer top-level EMAGE repo if present
    candidates.append(Path('EMAGE'))
    # Fallback to SadTalker vendor copy
    candidates.append(Path('SadTalker') / 'EMAGE')
    # De-dupe while preserving order
    seen = set()
    uniq: list[Path] = []
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def resolve_emage_root() -> Optional[Path]:
    """Pick the best available EMAGE root path, or None if unavailable."""
    for root in _candidate_emage_paths():
        try:
            if root.exists() and (root / 'models').exists():
                return root
        except Exception:
            continue
    return None


def emage_available() -> bool:
    """Check if EMAGE code and weights exist locally."""
    try:
        root = resolve_emage_root()
        if not root:
            return False
        models_exist = (root / "models").exists()
        weights_exist = (root / "checkpoints" / "emage_best.pth").exists()
        return bool(models_exist and weights_exist)
    except Exception:
        return False


class EMageRealistic:
    def __init__(self):
        self.root = resolve_emage_root()
        self.available = emage_available()

    def generate_full_video(self, **kwargs):
        """Placeholder for EMAGE full-body generation.

        Currently returns None unless the EMAGE repo and weights are present.
        """
        if not self.available:
            raise Exception("EMAGE not available")
        # Integration point: call into EMAGE when wired up
        return None
