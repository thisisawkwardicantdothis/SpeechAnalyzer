import argparse
import shutil
import sys
from pathlib import Path

from rich.console import Console

from analyzers import ALL_MODULES
from transcriber import transcribe
from pipeline import run_pipeline

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze.py",
        description="SpeechAnalyzer — Local video transcription and text analysis",
    )
    parser.add_argument("video", help="Path to video file (MP4, MKV, ...)")
    parser.add_argument(
        "--modules",
        default="all",
        help=f"Comma-separated modules or 'all'. Available: {', '.join(ALL_MODULES)}",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Base output directory (default: output/)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    return parser


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        console.print("[bold red]ERROR:[/] ffmpeg not found.")
        console.print("  Install from: https://ffmpeg.org/download.html")
        sys.exit(1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _check_ffmpeg()

    video_path = Path(args.video)
    if not video_path.exists():
        console.print(f"[bold red]ERROR:[/] File not found: {video_path}")
        sys.exit(1)

    if args.modules == "all":
        modules = ALL_MODULES
    else:
        modules = [m.strip() for m in args.modules.split(",")]
        unknown = [m for m in modules if m not in ALL_MODULES]
        if unknown:
            console.print(f"[bold red]ERROR:[/] Unknown modules: {', '.join(unknown)}")
            console.print(f"  Available: {', '.join(ALL_MODULES)}")
            sys.exit(1)

    output_dir = Path(args.output) / video_path.stem

    console.print(f"[bold]Transcribing[/] {video_path.name} (model: {args.whisper_model})...")
    segments, language = transcribe(str(video_path), args.whisper_model)
    console.print(f"  Detected language: [cyan]{language}[/]  |  Segments: {len(segments)}")

    console.print(f"[bold]Analysing[/] modules: {', '.join(modules)}")
    run_pipeline(
        segments=segments,
        language=language,
        modules=modules,
        output_dir=str(output_dir),
    )

    console.print(f"\n[bold green]Done![/] Results in: [underline]{output_dir}[/]")


if __name__ == "__main__":
    main()
