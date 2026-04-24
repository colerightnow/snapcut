import json
import subprocess
from pathlib import Path

from .models import MediaFile

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mxf", ".m4v", ".avi", ".mkv", ".webm",
    ".MP4", ".MOV", ".MXF", ".M4V", ".AVI", ".MKV", ".WEBM",
}

AUDIO_EXTENSIONS = {
    ".m4a", ".wav", ".aif", ".aiff", ".mp3", ".flac",
    ".M4A", ".WAV", ".AIF", ".AIFF", ".MP3", ".FLAC",
}


def find_media_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in VIDEO_EXTENSIONS)


def find_audio_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in AUDIO_EXTENSIONS)


def probe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def get_duration(path: Path) -> float:
    info = probe(path)
    return float(info["format"]["duration"])


def load_media_file(path: Path) -> MediaFile:
    info = probe(path)
    duration = float(info["format"]["duration"])
    video = next((s for s in info.get("streams", []) if s["codec_type"] == "video"), None)

    if video:
        width = int(video.get("width", 1920))
        height = int(video.get("height", 1080))
        num, den = video.get("r_frame_rate", "24/1").split("/")
        frame_rate = int(num) / int(den)
    else:
        width, height, frame_rate = 1920, 1080, 23.976

    return MediaFile(path=path, duration=duration, width=width, height=height, frame_rate=frame_rate)


def extract_audio(video_path: Path, output_path: Path, sample_rate: int = 16000) -> Path:
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-y", str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def extract_audio_from_audio(audio_path: Path, output_path: Path, sample_rate: int = 16000) -> Path:
    """Re-encode audio file to mono WAV for analysis."""
    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-y", str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
