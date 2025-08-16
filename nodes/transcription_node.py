import os
import json
import subprocess
import tempfile
import whisper
import datetime
from pathlib import Path
from langchain_core.tools import tool

from Classes.agent_state import AgentState

AUDIO_DIR = Path(os.environ.get("SERMON_AUDIO_DIR", r"C:\\Users\\Videos"))
DEFAULT_MODEL = "small.en" # We chose small.en for beter reco. Base has trouble with recognising words sometimes.

def _find_latest_media(path: Path) -> Path | None:
    """Find the latest media file in the given directory."""
    exts = (".mp3", ".mp4", ".wav", ".m4a", ".mov")
    if not path.exists():
        return None
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _validate_media_file(file_path: str) -> Path:
    """Validate that the provided file exists and is a supported media format."""
    supported_exts = (".mp3", ".mp4", ".wav", ".m4a", ".mov")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: '{file_path}'")

    if not path.is_file():
        raise ValueError(f"Path is not a file: '{file_path}'")

    if path.suffix.lower() not in supported_exts:
        raise ValueError(f"Unsupported file format: '{path.suffix}'. Supported formats: {supported_exts}")

    return path

def _extract_audio_if_needed(file_path: Path) -> Path:
    """Extract audio from video file if needed, return path to audio file."""
    video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    audio_exts = (".mp3", ".wav", ".m4a", ".flac", ".ogg")

    # If it's already an audio file, return as-is
    if file_path.suffix.lower() in audio_exts:
        print(f"File is already audio format: {file_path.suffix}")
        return file_path

    # If it's a video file, extract audio
    if file_path.suffix.lower() in video_exts:
        print(f"Video file detected ({file_path.suffix}), extracting audio...")

        # Create output audio file path (same directory, .wav extension)
        audio_path = file_path.with_suffix('.wav')

        # Skip extraction if audio file already exists and is newer than video
        if audio_path.exists() and audio_path.stat().st_mtime > file_path.stat().st_mtime:
            print(f"Audio file already exists and is up-to-date: {audio_path}")
            return audio_path

        # Use FFmpeg to extract audio
        try:
            cmd = [
                'ffmpeg',
                '-i', str(file_path),           # Input video file
                '-vn',                          # No video
                '-acodec', 'pcm_s16le',         # Audio codec (uncompressed WAV)
                '-ar', '48000',                 # Sample rate (48kHz is what the audio file was recorded at)
                '-ac', '1',                     # Mono audio
                '-y',                           # Overwrite output file
                str(audio_path)                 # Output audio file
            ]

            print(f"Running FFmpeg: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if audio_path.exists():
                print(f"Audio extracted successfully: {audio_path}")
                print(f"Audio file size: {audio_path.stat().st_size / (1024*1024):.1f} MB")
                return audio_path
            else:
                raise RuntimeError("FFmpeg completed but audio file was not created")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg failed to extract audio. Error: {e.stderr}\n"
                f"Command: {' '.join(cmd)}\n"
                f"Make sure FFmpeg is installed and in your PATH."
            ) from e
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg is required to extract audio from video files but was not found. "
                "Please install FFmpeg and add it to your system PATH. "
                "Download from: https://ffmpeg.org/download.html"
            )

    # If it's neither audio nor video, return as-is and let Whisper handle it
    print(f"Unknown file type ({file_path.suffix}), passing to Whisper directly...")
    return file_path


def _format_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"

@tool
def transcribe_audio(state: AgentState):
    """
    CPU-only Whisper transcription with segment timestamps.

    Looks for state['filePath'] if provided; otherwise uses the latest media file in the given folder.

    Writes:
    - transcription.txt (full text)
    - transcription_segments.json (segments with start/end/text)

    Returns a short JSON string summary with keys: {"file", "model", "text_path", "segments_path"}.
    """
    # Resolve inputs with priority: CLI arg > AgentState > auto-detection
    file_path = None

    # 1. Check for CLI-provided file path via environment variable
    cli_file_path = os.environ.get("SERMON_FILE_PATH")
    if cli_file_path:
        file_path = str(_validate_media_file(cli_file_path))
        print(f"Using CLI-provided file: {file_path}")
    else:
        # 2. Check AgentState for compatibility
        try:
            state_file_path = state.get("filePath")
            if state_file_path:
                file_path = str(_validate_media_file(state_file_path))
                print(f"Using AgentState file: {file_path}")
        except Exception:
            pass

        # 3. Auto-detect latest file if no specific path provided
        if not file_path:
            latest = _find_latest_media(AUDIO_DIR)
            if latest is None:
                raise FileNotFoundError(
                    f"No media files found in {AUDIO_DIR}. "
                    f"Provide a file path via --file argument or set AgentState.filePath."
                )
            file_path = str(latest)
            print(f"Auto-detected latest file: {file_path}")

    model_name = os.environ.get("WHISPER_MODEL", DEFAULT_MODEL)
    output_txt = Path("transcription.txt")
    output_json = Path("transcription_segments.json")

    # Extract audio from video if needed
    audio_file_path = _extract_audio_if_needed(Path(file_path))
    file_path = str(audio_file_path)  # Update file_path to point to audio file

    # Ensure CPU-only and use multi-core where possible
    try:
        import torch

        torch.set_num_threads(max(1, os.cpu_count() or 1))
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, os.cpu_count() or 1)))
        os.environ.setdefault("MKL_NUM_THREADS", str(max(1, os.cpu_count() or 1)))
        device = "cpu"
    except Exception:
        device = "cpu"

    # File validation already done by _validate_media_file() above

    # Load Whisper model (CPU)
    model = whisper.load_model(model_name, device=device)

    # Measure transcription time
    start = datetime.datetime.now()

    # Transcribe audio (disable fp16 for CPU)
    print(f"Starting Whisper transcription of: {file_path}")
    result = model.transcribe(file_path, fp16=False, language="English")

    # Persist outputs
    output_txt.write_text(result.get("text", "").strip(), encoding="utf-8")

    segments_out = []
    for seg in result.get("segments", []) or []:
        segments_out.append(
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "start_str": _format_ts(float(seg.get("start", 0.0))),
                "end_str": _format_ts(float(seg.get("end", 0.0))),
                "text": (seg.get("text") or "").strip(),
            }
        )
    output_json.write_text(json.dumps({"file": file_path, "segments": segments_out}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print execution time
    end = datetime.datetime.now()
    delta = end - start
    minutes, seconds = divmod(delta.seconds, 60)
    print(f"Transcription completed in {int(minutes)}m {int(seconds)}s")

    summary = {
        "file": file_path,
        "model": model_name,
        "text_path": str(output_txt.resolve()),
        "segments_path": str(output_json.resolve()),
    }

    return json.dumps(summary)
