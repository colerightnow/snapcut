from pathlib import Path

from .models import Segment


def transcribe(
    audio_path: Path,
    api_key: str | None = None,
    model: str = "whisper-1",
    language: str | None = None,
) -> list[Segment]:
    """Transcribe audio with OpenAI Whisper API; returns timestamped segments."""
    from openai import OpenAI  # deferred

    client = OpenAI(api_key=api_key)

    kwargs: dict = dict(
        model=model,
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )
    if language:
        kwargs["language"] = language

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(file=f, **kwargs)

    return [
        Segment(
            start=float(seg.start),
            end=float(seg.end),
            text=seg.text.strip(),
        )
        for seg in response.segments
        if seg.text.strip()
    ]
