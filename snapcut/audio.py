import json
import subprocess
from pathlib import Path

from .models import MediaFile

SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".mxf", ".m4v", ".avi", ".mkv", ".webm",
    ".MP4", ".MOV", ".MXF", ".M4V", ".AVI", ".MKV", ".WEBM",
}


def find_media_files(folder: Path) -> list[Path]:
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix in SUPPORTED_EXTENSIONS
    )
    return files


def probe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def load_media_file(path: Path) -> MediaFile:
    info = probe(path)
    duration = float(info["format"]["duration"])

    video = next((s for s in info.get("streams", []) if s["codec_type"] == "video"), None)

    if video:
        width = int(video.get("width", 1920))
        height = int(video.get("height", 1080))
        num, den = video.get("r_frame_rate", "30/1").split("/")
        frame_rate = int(num) / int(den)
    else:
        width, height, frame_rate = 1920, 1080, 29.97

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
