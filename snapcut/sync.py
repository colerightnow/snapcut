"""Audio sync via cross-correlation — finds lav offset relative to camera audio."""
from pathlib import Path

import numpy as np


def find_offset(reference_path: Path, query_path: Path, sr: int = 8000) -> float:
    """
    Return the time offset (seconds) of query relative to reference.

    Positive  → query starts this many seconds AFTER reference begins.
    Negative  → query started this many seconds BEFORE reference begins.

    Uses FFT-based cross-correlation at a low sample rate for speed.
    """
    import librosa
    from scipy.signal import correlate

    ref, _ = librosa.load(str(reference_path), sr=sr, mono=True)
    qry, _ = librosa.load(str(query_path), sr=sr, mono=True)

    # Normalise
    ref = ref / (np.max(np.abs(ref)) + 1e-8)
    qry = qry / (np.max(np.abs(qry)) + 1e-8)

    corr = correlate(ref, qry, mode="full", method="fft")
    lag = int(np.argmax(np.abs(corr))) - (len(qry) - 1)

    return float(lag) / sr
