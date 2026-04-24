"""GPT-4 moment scoring — Phase 2."""
import json

from .models import Segment

_PROMPT = """\
You are a social media video editor who cuts 90-second clips for Instagram, TikTok, and YouTube Shorts.

Score each transcript segment 0–10:
  8–10  Strong hook, high energy, quotable, stops scrolling
  5–7   Solid content, worth including
  0–4   Filler, dead air, transition, repetition

Segments:
{transcript}

Respond with a JSON object:
{{"scores": [{{"start": float, "end": float, "score": float, "reason": "one sentence"}}]}}
"""


def score_segments(
    segments: list[Segment],
    api_key: str | None = None,
    model: str = "gpt-4o",
    threshold: float = 6.0,
) -> list[Segment]:
    """Return segments scoring >= threshold, with confidence set to score/10."""
    if not segments:
        return []

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    transcript = "\n".join(
        f"[{s.start:.1f}s-{s.end:.1f}s] {s.text}" for s in segments
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _PROMPT.format(transcript=transcript)}],
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    results: list[Segment] = []

    for item in data.get("scores", []):
        score = float(item.get("score", 0))
        if score < threshold:
            continue
        # Match back to original segment (within 1 second tolerance)
        for seg in segments:
            if abs(seg.start - item["start"]) < 1.0 and abs(seg.end - item["end"]) < 1.0:
                results.append(
                    Segment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        confidence=score / 10.0,
                    )
                )
                break

    return sorted(results, key=lambda s: s.start)
