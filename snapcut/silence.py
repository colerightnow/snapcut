from pathlib import Path

from .models import Segment


def detect_audio_segments(
    audio_path: Path,
    top_db: float = 40.0,
    min_silence_duration: float = 0.5,
    min_speech_duration: float = 1.0,
) -> list[Segment]:
    """Return non-silent regions using librosa energy analysis."""
    import librosa  # deferred — heavy import

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    intervals = librosa.effects.split(
        y,
        top_db=top_db,
        frame_length=2048,
        hop_length=512,
    )

    if len(intervals) == 0:
        return []

    raw = [(float(s) / sr, float(e) / sr) for s, e in intervals]

    # Merge intervals whose gap is smaller than min_silence_duration
    merged: list[tuple[float, float]] = [raw[0]]
    for start, end in raw[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < min_silence_duration:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return [
        Segment(start=s, end=e)
        for s, e in merged
        if e - s >= min_speech_duration
    ]
