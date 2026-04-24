import hashlib
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from .models import MediaFile, Segment

# (rounded fps) -> (format_name_suffix, frame_duration)
_FPS_TABLE: dict[float, tuple[str, str]] = {
    23.976: ("2398", "1001/24000s"),
    24.0:   ("24",   "100/2400s"),
    25.0:   ("25",   "100/2500s"),
    29.97:  ("2997", "1001/30000s"),
    30.0:   ("30",   "100/3000s"),
    50.0:   ("50",   "100/5000s"),
    59.94:  ("5994", "1001/60000s"),
    60.0:   ("60",   "100/6000s"),
}

_DROP_FRAME_RATES = {29.97, 59.94}


def _round_fps(fps: float) -> float:
    for known in _FPS_TABLE:
        if abs(fps - known) < 0.05:
            return known
    return round(fps, 3)


def _format_name(mf: MediaFile) -> str:
    fps = _round_fps(mf.frame_rate)
    suffix = _FPS_TABLE.get(fps, (str(int(fps)), ""))[0]
    return f"FFVideoFormat{mf.height}p{suffix}"


def _frame_duration(mf: MediaFile) -> str:
    fps = _round_fps(mf.frame_rate)
    if fps in _FPS_TABLE:
        return _FPS_TABLE[fps][1]
    f = Fraction(1, int(fps))
    return f"{f.numerator}/{f.denominator}s"


def _tc_format(mf: MediaFile) -> str:
    return "DF" if _round_fps(mf.frame_rate) in _DROP_FRAME_RATES else "NDF"


def _secs(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    f = Fraction(seconds).limit_denominator(1_000_000)
    return f"{f.numerator}s" if f.denominator == 1 else f"{f.numerator}/{f.denominator}s"


def _uid(path: Path) -> str:
    h = hashlib.md5(str(path.absolute()).encode()).hexdigest().upper()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def _indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = pad + "  "
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def _build_spine(
    event: ET.Element,
    project_name: str,
    fmt_ref: str,
    tc_format: str,
    clips: list[tuple[str, str, float, float]],  # (asset_id, clip_name, src_start, duration)
) -> None:
    """Add a project+sequence+spine to the event."""
    total = sum(d for _, _, _, d in clips)
    project = ET.SubElement(event, "project", name=project_name)
    sequence = ET.SubElement(
        project, "sequence",
        format=fmt_ref,
        duration=_secs(total),
        tcStart="0s",
        tcFormat=tc_format,
        audioLayout="stereo",
        audioRate="48k",
    )
    spine = ET.SubElement(sequence, "spine")
    offset = 0.0
    for asset_id, clip_name, src_start, duration in clips:
        ET.SubElement(
            spine, "asset-clip",
            name=clip_name,
            ref=asset_id,
            offset=_secs(offset),
            start=_secs(src_start),
            duration=_secs(duration),
            tcFormat=tc_format,
        )
        offset += duration


def build_fcpxml(media_files: list[MediaFile], output_dir: Path) -> Path:
    root = ET.Element("fcpxml", version="1.11")
    resources = ET.SubElement(root, "resources")
    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name="SnapCut")

    # Deduplicate formats
    seen_formats: dict[tuple, str] = {}
    asset_ids: dict[Path, str] = {}
    next_id = 1

    primary_mf = media_files[0]

    for mf in media_files:
        fmt_key = (mf.width, mf.height, _round_fps(mf.frame_rate))
        if fmt_key not in seen_formats:
            fmt_id = f"r{next_id}"
            next_id += 1
            seen_formats[fmt_key] = fmt_id
            ET.SubElement(
                resources, "format",
                id=fmt_id,
                name=_format_name(mf),
                frameDuration=_frame_duration(mf),
                width=str(mf.width),
                height=str(mf.height),
            )

        asset_id = f"r{next_id}"
        next_id += 1
        asset_ids[mf.path] = asset_id

        asset = ET.SubElement(
            resources, "asset",
            id=asset_id,
            name=mf.path.stem,
            uid=_uid(mf.path),
            start="0s",
            duration=_secs(mf.duration),
            hasVideo="1",
            hasAudio="1",
        )
        ET.SubElement(asset, "media-rep", kind="original-media", src=mf.path.absolute().as_uri())

    fmt_key = (primary_mf.width, primary_mf.height, _round_fps(primary_mf.frame_rate))
    fmt_ref = seen_formats[fmt_key]
    tc_fmt = _tc_format(primary_mf)

    # All Footage — full clips in order
    all_clips = [
        (asset_ids[mf.path], mf.path.stem, 0.0, mf.duration)
        for mf in media_files
    ]
    _build_spine(event, "All Footage", fmt_ref, tc_fmt, all_clips)

    # Selects — librosa non-silent regions
    select_clips: list[tuple[str, str, float, float]] = []
    for mf in media_files:
        for i, seg in enumerate(mf.select_segments):
            select_clips.append((
                asset_ids[mf.path],
                f"{mf.path.stem}_s{i + 1}",
                seg.start,
                seg.duration,
            ))
    _build_spine(event, "Selects", fmt_ref, tc_fmt, select_clips or all_clips)

    # Edit — Whisper speech regions (tighter than Selects)
    edit_clips: list[tuple[str, str, float, float]] = []
    for mf in media_files:
        for i, seg in enumerate(mf.edit_segments):
            edit_clips.append((
                asset_ids[mf.path],
                f"{mf.path.stem}_e{i + 1}",
                seg.start,
                seg.duration,
            ))
    _build_spine(event, "Edit", fmt_ref, tc_fmt, edit_clips or all_clips)

    _indent(root)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    output = output_dir / "SnapCut.fcpxml"
    with open(output, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)
    return output
