# Sermon Shorts Setup Guide

## Prerequisites

### 1. Install FFmpeg
FFmpeg is required for Whisper to process video files.

**Windows:**
1. Download FFmpeg from: https://ffmpeg.org/download.html
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your system PATH:
   - Open System Properties → Advanced → Environment Variables
   - Edit the PATH variable and add the FFmpeg bin directory
   - Restart your command prompt/terminal

**Alternative (using Chocolatey):**
```bash
choco install ffmpeg
```

**Alternative (using winget):**
```bash
winget install Gyan.FFmpeg
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
SERMON_AUDIO_DIR=G:\Thrive\Sermon Videos\Audio Files\
WHISPER_MODEL=base.en
```

## Usage

### Basic Usage (Auto-detect latest file)
```bash
python agent.py
```

### Specify a specific file
```bash
python agent.py --file "path/to/sermon.mp4"
python agent.py -f "C:\path\to\sermon.mp3"
```

### Example with your file
```bash
python agent.py --file "S:\2025-08-10_10-25-50.mp4"
```

## Output Files
The agent will create:
- `transcription.txt` - Full sermon transcript
- `transcription_segments.json` - Transcript with timestamps
- `recommendations.json` - Clip recommendations (machine-readable)
- `recommendations.txt` - Clip recommendations (human-readable)

## Troubleshooting

### "FFmpeg not found" error
- Ensure FFmpeg is installed and in your PATH
- Restart your terminal after installing FFmpeg
- Test with: `ffmpeg -version`

### "No module named 'whisper'" error
- Run: `pip install openai-whisper`

### OpenAI API errors
- Verify your API key is set correctly
- Check your OpenAI account has sufficient credits

### File not found errors
- Ensure the file path is correct
- Use forward slashes or raw strings for Windows paths
- Check file permissions

## Performance Notes
- Large video files (like your 2.6GB sermon) will take significant time to transcribe
- The base.en model is optimized for speed vs accuracy
- Use the "small" model for better accuracy: set `WHISPER_MODEL=small` in .env
- Transcription is CPU-only and will use all available cores
