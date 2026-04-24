import tempfile
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .audio import (
    find_media_files, find_audio_files,
    load_media_file, extract_audio, extract_audio_from_audio, get_duration,
)
from .models import MediaFile, SyncedAudio
from .premiere_xml import build_premiere_xml
from .silence import detect_audio_segments
from .sync import find_offset
from .transcribe import transcribe
from .score import score_segments

console = Console()


@click.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, path_type=Path),
              help="Output directory [default: FOLDER]")
@click.option("--silence-threshold", default=40.0, show_default=True,
              help="Silence threshold — dB below peak considered silent")
@click.option("--min-silence", default=0.5, show_default=True,
              help="Minimum silence gap (s) before creating a cut")
@click.option("--min-speech", default=1.0, show_default=True,
              help="Minimum segment duration (s) to keep")
@click.option("--language", default=None,
              help="Language hint for Whisper (e.g. 'en')")
@click.option("--skip-transcription", is_flag=True,
              help="Skip Whisper transcription")
@click.option("--skip-scoring", is_flag=True,
              help="Skip GPT-4 moment scoring (Selects will use Whisper or librosa)")
@click.option("--skip-sync", is_flag=True,
              help="Skip lav mic sync")
@click.option("--score-threshold", default=6.0, show_default=True,
              help="Minimum GPT-4 score (0–10) to include in Selects")
@click.option("--api-key", envvar="OPENAI_API_KEY", default=None,
              help="OpenAI API key [env: OPENAI_API_KEY]")
@click.version_option()
def main(
    folder: Path,
    output_dir: Path | None,
    silence_threshold: float,
    min_silence: float,
    min_speech: float,
    language: str | None,
    skip_transcription: bool,
    skip_scoring: bool,
    skip_sync: bool,
    score_threshold: float,
    api_key: str | None,
) -> None:
    """SnapCut — footage folder → Premiere XML (All Footage / Selects / Edit).

    FOLDER is scanned for video files (mp4, mov, mxf, m4v, mkv, webm)
    and audio files (m4a, wav, aif). Audio files are synced to video
    via cross-correlation.
    """
    output_dir = output_dir or folder
    output_dir.mkdir(parents=True, exist_ok=True)

    needs_api = not skip_transcription or not skip_scoring
    if needs_api and not api_key:
        raise click.UsageError(
            "OPENAI_API_KEY not set. Pass --api-key or set the env var, "
            "or use --skip-transcription --skip-scoring."
        )

    # ── Discover files ────────────────────────────────────────────────────────
    video_paths = find_media_files(folder)
    audio_paths = find_audio_files(folder)

    if not video_paths:
        raise click.ClickException(f"No video files found in {folder}")

    console.print(
        f"\n[bold]SnapCut[/bold]  "
        f"[cyan]{len(video_paths)} video[/cyan]  "
        f"[green]{len(audio_paths)} audio[/green]  "
        f"→  [yellow]{output_dir}[/yellow]\n"
    )

    media_files: list[MediaFile] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:

            # ── Per-video processing ──────────────────────────────────────────
            vtask = progress.add_task("Videos", total=len(video_paths))

            for vpath in video_paths:
                progress.update(vtask, description=f"Probe    {vpath.name}")
                mf = load_media_file(vpath)

                cam_wav = tmp_path / f"{vpath.stem}_cam.wav"
                progress.update(vtask, description=f"Audio    {vpath.name}")
                extract_audio(vpath, cam_wav)

                # Silence detection on camera audio
                progress.update(vtask, description=f"Silence  {vpath.name}")
                mf.audio_segments = detect_audio_segments(
                    cam_wav,
                    top_db=silence_threshold,
                    min_silence_duration=min_silence,
                    min_speech_duration=min_speech,
                )

                # Whisper transcription
                if not skip_transcription:
                    progress.update(vtask, description=f"Whisper  {vpath.name}")
                    try:
                        mf.speech_segments = transcribe(cam_wav, api_key=api_key, language=language)
                    except Exception as exc:
                        console.print(f"[yellow]Whisper error ({vpath.name}): {exc}[/yellow]")

                # GPT-4 scoring
                if not skip_scoring and mf.speech_segments:
                    progress.update(vtask, description=f"Scoring  {vpath.name}")
                    try:
                        mf.scored_segments = score_segments(
                            mf.speech_segments,
                            api_key=api_key,
                            threshold=score_threshold,
                        )
                    except Exception as exc:
                        console.print(f"[yellow]Scoring error ({vpath.name}): {exc}[/yellow]")

                # Lav sync
                if not skip_sync and audio_paths:
                    progress.update(vtask, description=f"Sync     {vpath.name}")
                    for apath in audio_paths:
                        try:
                            lav_wav = tmp_path / f"{apath.stem}_lav.wav"
                            extract_audio_from_audio(apath, lav_wav)
                            offset = find_offset(cam_wav, lav_wav)
                            lav_dur = get_duration(apath)
                            mf.synced_audio.append(
                                SyncedAudio(path=apath, duration=lav_dur, offset=offset)
                            )
                            console.print(
                                f"  [dim]sync[/dim]  {apath.name}  "
                                f"[cyan]offset {offset:+.2f}s[/cyan]"
                            )
                        except Exception as exc:
                            console.print(f"[yellow]Sync error ({apath.name}): {exc}[/yellow]")

                media_files.append(mf)
                progress.advance(vtask)

    # ── Export ────────────────────────────────────────────────────────────────
    console.print("\nBuilding Premiere XML…")
    output = build_premiere_xml(media_files, output_dir)
    console.print(f"[green]Done:[/green]  {output}\n")

    total   = sum(mf.duration for mf in media_files)
    sel_dur = sum(s.duration for mf in media_files for s in mf.select_segments)
    edit_dur = sum(s.duration for mf in media_files for s in mf.edit_segments)

    console.print(f"  All Footage  {_fmt(total)}")
    console.print(f"  Selects      {_fmt(sel_dur)}  ({sel_dur/total*100:.0f}%)")
    console.print(f"  Edit         {_fmt(edit_dur)}  ({edit_dur/total*100:.0f}%)")
    console.print()


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
