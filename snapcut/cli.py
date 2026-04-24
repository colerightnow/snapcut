import tempfile
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .audio import find_media_files, load_media_file, extract_audio
from .fcpxml import build_fcpxml
from .models import MediaFile
from .silence import detect_audio_segments
from .transcribe import transcribe

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
              help="Minimum segment duration (s) to include in Selects/Edit")
@click.option("--language", default=None,
              help="Language hint for Whisper (e.g. 'en')")
@click.option("--skip-transcription", is_flag=True,
              help="Skip Whisper — Selects and Edit will use librosa only")
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
    api_key: str | None,
) -> None:
    """SnapCut: turn a footage folder into three FCPXML sequences.

    FOLDER is scanned for video files (mp4, mov, mxf, m4v, avi, mkv, webm).
    Outputs SnapCut.fcpxml with three projects: All Footage, Selects, Edit.
    """
    output_dir = output_dir or folder
    output_dir.mkdir(parents=True, exist_ok=True)

    if not skip_transcription and not api_key:
        raise click.UsageError(
            "OPENAI_API_KEY not set. Pass --api-key or set the env var, "
            "or use --skip-transcription to run without Whisper."
        )

    # Discover files
    paths = find_media_files(folder)
    if not paths:
        raise click.ClickException(f"No supported video files found in {folder}")

    console.print(f"[bold]SnapCut[/bold] — found {len(paths)} file(s) in [cyan]{folder}[/cyan]")

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
            task = progress.add_task("Processing", total=len(paths))

            for path in paths:
                progress.update(task, description=f"Probing  {path.name}")
                mf = load_media_file(path)

                # Extract audio for analysis
                wav = tmp_path / f"{path.stem}.wav"
                progress.update(task, description=f"Audio    {path.name}")
                extract_audio(path, wav)

                # Silence detection
                progress.update(task, description=f"Silence  {path.name}")
                mf.audio_segments = detect_audio_segments(
                    wav,
                    top_db=silence_threshold,
                    min_silence_duration=min_silence,
                    min_speech_duration=min_speech,
                )

                # Whisper transcription
                if not skip_transcription:
                    progress.update(task, description=f"Whisper  {path.name}")
                    try:
                        mf.speech_segments = transcribe(wav, api_key=api_key, language=language)
                    except Exception as exc:
                        console.print(f"[yellow]Whisper failed for {path.name}: {exc}[/yellow]")

                media_files.append(mf)
                progress.advance(task)

    # FCPXML
    console.print("Building FCPXML…")
    output = build_fcpxml(media_files, output_dir)
    console.print(f"[green]Done:[/green] {output}")

    # Summary table
    total_dur = sum(mf.duration for mf in media_files)
    select_dur = sum(s.duration for mf in media_files for s in mf.select_segments)
    edit_dur = sum(s.duration for mf in media_files for s in mf.edit_segments)

    console.print()
    console.print(f"  All Footage  {_fmt(total_dur)}")
    console.print(f"  Selects      {_fmt(select_dur)}  ({select_dur / total_dur * 100:.0f}%)")
    console.print(f"  Edit         {_fmt(edit_dur)}  ({edit_dur / total_dur * 100:.0f}%)")


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
