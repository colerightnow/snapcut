from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Segment:
    start: float  # seconds
    end: float    # seconds
    text: str = ""
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class MediaFile:
    path: Path
    duration: float
    width: int = 1920
    height: int = 1080
    frame_rate: float = 29.97
    audio_segments: list[Segment] = field(default_factory=list)   # librosa
    speech_segments: list[Segment] = field(default_factory=list)  # Whisper

    @property
    def select_segments(self) -> list[Segment]:
        """Non-silent regions from librosa; falls back to full clip."""
        return self.audio_segments or [Segment(0.0, self.duration)]

    @property
    def edit_segments(self) -> list[Segment]:
        """Speech regions from Whisper; falls back to audio_segments."""
        return self.speech_segments or self.audio_segments or [Segment(0.0, self.duration)]
