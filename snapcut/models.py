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
class SyncedAudio:
    path: Path
    duration: float
    offset: float  # seconds — positive: lav starts after video; negative: before


@dataclass
class MediaFile:
    path: Path
    duration: float
    width: int = 1920
    height: int = 1080
    frame_rate: float = 23.976
    audio_segments: list[Segment] = field(default_factory=list)   # librosa
    speech_segments: list[Segment] = field(default_factory=list)  # Whisper
    scored_segments: list[Segment] = field(default_factory=list)  # GPT-4
    synced_audio: list[SyncedAudio] = field(default_factory=list) # lav mics

    @property
    def select_segments(self) -> list[Segment]:
        """GPT-4 scored moments → Selects. Falls back to speech → audio → full clip."""
        return (
            self.scored_segments
            or self.speech_segments
            or self.audio_segments
            or [Segment(0.0, self.duration)]
        )

    @property
    def edit_segments(self) -> list[Segment]:
        """Non-silent regions → Edit. Falls back to full clip."""
        return self.audio_segments or [Segment(0.0, self.duration)]
