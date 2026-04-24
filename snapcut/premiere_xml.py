"""Premiere-compatible XML export (FCP7 / XMEML format)."""
import xml.etree.ElementTree as ET
from pathlib import Path

from .models import MediaFile, Segment, SyncedAudio


# ── Frame rate helpers ────────────────────────────────────────────────────────

def _fps_info(fps: float) -> tuple[int, bool]:
    """Return (timebase, is_ntsc)."""
    for known, tb, ntsc in [
        (23.976, 24, True),
        (29.97,  30, True),
        (59.94,  60, True),
        (24.0,   24, False),
        (25.0,   25, False),
        (50.0,   50, False),
        (60.0,   60, False),
    ]:
        if abs(fps - known) < 0.05:
            return tb, ntsc
    return round(fps), False


def _to_frames(seconds: float, fps: float) -> int:
    tb, ntsc = _fps_info(fps)
    if ntsc:
        return round(seconds * tb * 1000 / 1001)
    return round(seconds * tb)


def _add_rate(parent: ET.Element, fps: float) -> None:
    tb, ntsc = _fps_info(fps)
    r = ET.SubElement(parent, "rate")
    ET.SubElement(r, "timebase").text = str(tb)
    ET.SubElement(r, "ntsc").text = "TRUE" if ntsc else "FALSE"


# ── File definitions ──────────────────────────────────────────────────────────

def _file_def(parent: ET.Element, fid: str, mf: MediaFile) -> ET.Element:
    f = ET.SubElement(parent, "file", id=fid)
    ET.SubElement(f, "name").text = mf.path.name
    ET.SubElement(f, "pathurl").text = mf.path.absolute().as_uri()
    _add_rate(f, mf.frame_rate)
    ET.SubElement(f, "duration").text = str(_to_frames(mf.duration, mf.frame_rate))
    med = ET.SubElement(f, "media")
    vid = ET.SubElement(med, "video")
    sc = ET.SubElement(vid, "samplecharacteristics")
    _add_rate(sc, mf.frame_rate)
    ET.SubElement(sc, "width").text = str(mf.width)
    ET.SubElement(sc, "height").text = str(mf.height)
    aud = ET.SubElement(med, "audio")
    asc = ET.SubElement(aud, "samplecharacteristics")
    ET.SubElement(asc, "depth").text = "16"
    ET.SubElement(asc, "samplerate").text = "48000"
    return f


def _audio_file_def(parent: ET.Element, fid: str, path: Path, duration_secs: float, fps: float) -> ET.Element:
    f = ET.SubElement(parent, "file", id=fid)
    ET.SubElement(f, "name").text = path.name
    ET.SubElement(f, "pathurl").text = path.absolute().as_uri()
    _add_rate(f, fps)
    ET.SubElement(f, "duration").text = str(_to_frames(duration_secs, fps))
    med = ET.SubElement(f, "media")
    aud = ET.SubElement(med, "audio")
    asc = ET.SubElement(aud, "samplecharacteristics")
    ET.SubElement(asc, "depth").text = "16"
    ET.SubElement(asc, "samplerate").text = "48000"
    return f


# ── Clip item builders ────────────────────────────────────────────────────────

def _video_clipitem(
    track: ET.Element,
    iid: str,
    fid: str,
    mf: MediaFile,
    tl_start: int,
    src_in: int,
    src_out: int,
    defined: set[str],
    links: list[str] | None = None,
) -> ET.Element:
    ci = ET.SubElement(track, "clipitem", id=iid)
    ET.SubElement(ci, "name").text = mf.path.stem
    ET.SubElement(ci, "duration").text = str(src_out - src_in)
    _add_rate(ci, mf.frame_rate)
    ET.SubElement(ci, "start").text = str(tl_start)
    ET.SubElement(ci, "end").text = str(tl_start + src_out - src_in)
    ET.SubElement(ci, "in").text = str(src_in)
    ET.SubElement(ci, "out").text = str(src_out)
    if fid not in defined:
        _file_def(ci, fid, mf)
        defined.add(fid)
    else:
        ET.SubElement(ci, "file", id=fid)
    if links:
        for lid in links:
            lk = ET.SubElement(ci, "link")
            ET.SubElement(lk, "linkclipref").text = lid
    return ci


def _audio_clipitem(
    track: ET.Element,
    iid: str,
    fid: str,
    fps: float,
    tl_start: int,
    src_in: int,
    src_out: int,
    track_index: int,
    name: str,
    defined: set[str],
    audio_path: Path | None = None,
    audio_dur: float | None = None,
    links: list[str] | None = None,
) -> ET.Element:
    ci = ET.SubElement(track, "clipitem", id=iid)
    ET.SubElement(ci, "name").text = name
    ET.SubElement(ci, "duration").text = str(src_out - src_in)
    _add_rate(ci, fps)
    ET.SubElement(ci, "start").text = str(tl_start)
    ET.SubElement(ci, "end").text = str(tl_start + src_out - src_in)
    ET.SubElement(ci, "in").text = str(src_in)
    ET.SubElement(ci, "out").text = str(src_out)
    if fid not in defined and audio_path and audio_dur is not None:
        _audio_file_def(ci, fid, audio_path, audio_dur, fps)
        defined.add(fid)
    else:
        ET.SubElement(ci, "file", id=fid)
    st = ET.SubElement(ci, "sourcetrack")
    ET.SubElement(st, "mediatype").text = "audio"
    ET.SubElement(st, "trackindex").text = str(track_index)
    if links:
        for lid in links:
            lk = ET.SubElement(ci, "link")
            ET.SubElement(lk, "linkclipref").text = lid
    return ci


# ── Sequence builder ──────────────────────────────────────────────────────────

def _build_sequence(
    seq_id: str,
    name: str,
    clips: list[tuple[MediaFile, float, float]],  # (mf, src_start_secs, duration_secs)
    defined: set[str],
    include_lav: bool = True,
) -> ET.Element:
    """
    Build a <sequence> element.
    clips = list of (MediaFile, src_start, duration) in timeline order.
    """
    if not clips:
        return ET.Element("sequence", id=seq_id)

    fps = clips[0][0].frame_rate
    total_frames = sum(_to_frames(dur, fps) for _, _, dur in clips)

    seq = ET.Element("sequence", id=seq_id)
    ET.SubElement(seq, "name").text = name
    ET.SubElement(seq, "duration").text = str(total_frames)
    _add_rate(seq, fps)

    tc = ET.SubElement(seq, "timecode")
    _add_rate(tc, fps)
    ET.SubElement(tc, "string").text = "00:00:00:00"
    ET.SubElement(tc, "frame").text = "0"
    ET.SubElement(tc, "displayformat").text = "NDF"

    media = ET.SubElement(seq, "media")

    # ── Video ─────────────────────────────────────────────────────────────────
    video_el = ET.SubElement(media, "video")
    fmt = ET.SubElement(video_el, "format")
    sc = ET.SubElement(fmt, "samplecharacteristics")
    _add_rate(sc, fps)
    mf0 = clips[0][0]
    ET.SubElement(sc, "width").text = str(mf0.width)
    ET.SubElement(sc, "height").text = str(mf0.height)

    v_track = ET.SubElement(video_el, "track")

    # ── Audio tracks ──────────────────────────────────────────────────────────
    audio_el = ET.SubElement(media, "audio")
    cam_track = ET.SubElement(audio_el, "track")   # track 1: camera audio

    # Collect unique lav files across all clips
    lav_paths: list[Path] = []
    if include_lav:
        for mf, _, _ in clips:
            for sa in mf.synced_audio:
                if sa.path not in lav_paths:
                    lav_paths.append(sa.path)
    lav_tracks = [ET.SubElement(audio_el, "track") for _ in lav_paths]

    # ── Place clips ───────────────────────────────────────────────────────────
    tl_offset = 0  # running timeline position in frames

    for clip_idx, (mf, src_start, duration) in enumerate(clips):
        fid = f"file-{mf.path.stem.replace(' ', '_')}"
        src_in  = _to_frames(src_start, fps)
        src_out = _to_frames(src_start + duration, fps)
        dur_f   = src_out - src_in

        vid_id = f"{seq_id}-v{clip_idx}"
        cam_id = f"{seq_id}-a{clip_idx}-cam"

        # Video clip
        _video_clipitem(
            v_track, vid_id, fid, mf,
            tl_start=tl_offset, src_in=src_in, src_out=src_out,
            defined=defined, links=[cam_id],
        )

        # Camera audio
        _audio_clipitem(
            cam_track, cam_id, fid, fps,
            tl_start=tl_offset, src_in=src_in, src_out=src_out,
            track_index=1, name=mf.path.stem,
            defined=defined, links=[vid_id],
        )

        # Lav audio tracks
        if include_lav:
            for li, lav_path in enumerate(lav_paths):
                # Find this lav's sync info for this mf
                sa = next((s for s in mf.synced_audio if s.path == lav_path), None)
                if sa is None:
                    continue

                lav_fid = f"file-{lav_path.stem.replace(' ', '_')}"
                lav_id  = f"{seq_id}-a{clip_idx}-lav{li}"

                offset_f = _to_frames(sa.offset, fps)

                if sa.offset >= 0:
                    # Lav starts after video: shift lav right in timeline
                    lav_tl_start = tl_offset + offset_f
                    lav_src_in   = src_in
                    lav_src_out  = min(
                        _to_frames(sa.duration, fps),
                        tl_offset + dur_f - lav_tl_start + lav_src_in,
                    )
                else:
                    # Lav started before video: trim lav head
                    lav_tl_start = tl_offset
                    lav_src_in   = src_in + abs(offset_f)
                    lav_src_out  = lav_src_in + dur_f

                if lav_src_out <= lav_src_in:
                    continue

                _audio_clipitem(
                    lav_tracks[li], lav_id, lav_fid, fps,
                    tl_start=lav_tl_start,
                    src_in=lav_src_in, src_out=lav_src_out,
                    track_index=1, name=lav_path.stem,
                    defined=defined,
                    audio_path=lav_path, audio_dur=sa.duration,
                )

        tl_offset += dur_f

    return seq


# ── Public entry point ────────────────────────────────────────────────────────

def build_premiere_xml(media_files: list[MediaFile], output_dir: Path) -> Path:
    defined: set[str] = set()

    # All Footage — full clips with synced lav
    all_clips = [(mf, 0.0, mf.duration) for mf in media_files]
    seq_all = _build_sequence("seq-all", "All Footage", all_clips, defined, include_lav=True)

    # Selects — GPT-4 scored (or speech) segments, lav included
    sel_clips = [
        (mf, seg.start, seg.duration)
        for mf in media_files
        for seg in mf.select_segments
    ]
    seq_sel = _build_sequence("seq-sel", "Selects", sel_clips or all_clips, defined, include_lav=True)

    # Edit — silence-cut segments, lav included
    edit_clips = [
        (mf, seg.start, seg.duration)
        for mf in media_files
        for seg in mf.edit_segments
    ]
    seq_edit = _build_sequence("seq-edit", "Edit", edit_clips or all_clips, defined, include_lav=True)

    # Wrap in bin
    root = ET.Element("xmeml", version="5")
    bin_el = ET.SubElement(root, "bin")
    ET.SubElement(bin_el, "name").text = "SnapCut"
    children = ET.SubElement(bin_el, "children")
    children.append(seq_all)
    children.append(seq_sel)
    children.append(seq_edit)

    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    output = output_dir / "SnapCut.xml"
    with open(output, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(b'<!DOCTYPE xmeml PUBLIC "-//Apple//DTD XMEML 1//EN" "http://www.apple.com/DTDs/XMEML1.dtd">\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)
    return output
