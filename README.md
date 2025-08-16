# Sermon-Shorts AI Agent

An intelligent tool that automatically analyzes sermon recordings and recommends the best 60-90 second clips for viral social media content (YouTube Shorts, TikTok, Instagram Reels). The agent takes away the tedium of manually searching through sermons to find the best moments for social media and recommends moments for us to edit clips from the sermon into social media content.

## ✨ What It Does

- **🎯 Smart Clip Selection**: Identifies engaging moments with pastor personality, humor, and relatable content
- **📝 Accurate Transcription**: Uses Whisper small.en model for high-quality speech-to-text
- **🤖 AI-Powered Analysis**: GPT-5-mini analyzes content for viral potential and social media optimization
- **⚡ Automated Workflow**: Processes video/audio files and outputs ready-to-use recommendations

### Future Enhancements

- **🤨 Embeddings for Semantic Analysis**: Using Embeddings for semantic analysis and filtering out noise we can filter out segments with low content quality scores or repetitive worship/music transcription errors - which should lead to better clip recommendations.
- **🎥 Video Clip Generation**: Automatically create video clips from recommended timestamps
- **🖼️ Image Generation**: Generate images for social media posts based on sermon content
- **🔗 Auto-Post to Social Media**: Integrate with social media APIs to automatically post clips and images

## Technology Stack

- **Speech-to-Text**: OpenAI Whisper (small.en model for optimal accuracy/speed balance)
- **AI Framework**: LangGraph with GPT-4o-mini
- **Audio Processing**: FFmpeg for video-to-audio conversion
- **Language**: Python 3.11+
- **Key Libraries**: langchain, openai-whisper, torch, numpy

## Quick Start

### Prerequisites
- Python 3.11 or higher
- FFmpeg installed and in PATH ([Download here](https://ffmpeg.org/download.html))
- OpenAI API key ([Get one here](https://platform.openai.com/account/api-keys))

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/ThriveCommunityChurch/Sermon-Shorts.git
   cd Sermon-Shorts
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   SERMON_AUDIO_DIR=C:\Users\Videos  # Optional: default directory for auto-detection
   ```

### Usage

**Auto-detect latest file in directory:**
```bash
python agent.py
```

**Specify a specific file:**
```bash
python agent.py --file "path/to/sermon.mp4"
python agent.py -f "C:\path\to\sermon.mp3"
```

**Example with full path:**
```bash
python agent.py --file "S:\2025-07-27_09-58-11.mp4"
```

## Output Files

The agent generates several files in the project directory:

- **`transcription.txt`** - Complete sermon transcript (human-readable)
- **`transcription_segments.json`** - Transcript with precise timestamps (machine-readable)
- **`recommendations.json`** - Clip recommendations with metadata (machine-readable)
- **`recommendations.txt`** - Clip recommendations summary (human-readable)

## How It Works

### 1. **Audio Processing** (`transcription_node.py`)
- Automatically detects and converts video files to audio using FFmpeg
- Uses Whisper `small.en` model for optimal accuracy/speed balance
- Generates timestamped transcript segments for precise clip boundaries

### 2. **AI Analysis** (`analysis_node.py`)
- GPT-4o-mini analyzes transcript for viral potential
- **Enhanced Content Detection**:
  - Pastor personality and humor moments
  - Contemporary references and relatable analogies
  - Emotional peaks and inspirational content
  - Community interactions and authentic moments
- **Smart Filtering**: Ignores pre-service chatter and background noise
- **Optimized Duration**: Targets 60-90 second clips for maximum engagement

### 3. **Intelligent Orchestration** (`agent.py`)
- LangGraph manages the workflow and tool coordination
- Handles file detection, processing, and error recovery
- Provides CLI interface for easy operation

## Recent Improvements

- **Upgraded to Whisper small.en** for better transcription accuracy
- **Enhanced AI prompt** for diverse content selection and viral optimization
- **Improved clip targeting** (60-90 seconds for optimal social media performance)
- **Better personality detection** (humor, vulnerability, relatability)
- **Contemporary reference prioritization** (smartphones, social media, current events)
- **Comprehensive test suite** with 25+ unit tests for reliability

## Example Output

```json
{
  "start_sec": 356.28,
  "end_sec": 392.44,
  "start": "05:56",
  "end": "06:32",
  "description": "Warm, inclusive greeting + on-stage announcement: 'Hey, church. Good morning, y'all... As you may have noticed, we got a new worship set up today... Laurel's birthday — be sure to embarrass her.' Fun, human, community-oriented moment perfect for social intros and reels.",
  "confidence": 0.88,
  "reasoning": "A friendly address to the audience with a humorous, humanizing call-to-action (embarrass her). Short, standalone, and showcases church culture and pastor charisma — great for shareability and engagement."
}
```

## Testing

Run the comprehensive test suite:
```bash
python test_cli.py
```

The test suite includes:
- File validation and audio processing tests
- Transcription accuracy verification
- AI analysis functionality tests
- Configuration and system integration tests

## 🔧 Troubleshooting

### Common Issues

**"FFmpeg not found" error**
```bash
# Windows (using Chocolatey)
choco install ffmpeg

# Windows (using winget)
winget install Gyan.FFmpeg

# Verify installation
ffmpeg -version
```

**"No module named 'whisper'" error**
```bash
pip install -r requirements.txt
```

**OpenAI API errors**
- Verify your API key is set correctly in `.env`
- Check your OpenAI account has sufficient credits
- Ensure you have access to GPT-4o-mini model

**Transcription quality issues**
- The system uses `small.en` model for balanced accuracy/speed
- For higher accuracy, you can modify `DEFAULT_MODEL` in `transcription_node.py`
- Supported models: `tiny.en`, `base.en`, `small.en`, `medium.en`, `large`

## Configuration

### Environment Variables
- `OPENAI_API_KEY` - Required: Your OpenAI API key
- `SERMON_AUDIO_DIR` - Optional: Default directory for auto-detection (default: `C:\Users\Videos`)

### Supported File Formats
- **Audio**: `.mp3`, `.wav`, `.m4a`
- **Video**: `.mp4`, `.mov` (automatically converted to audio)

## Performance

### Typical Processing Times
- **1.5-hour sermon**: ~8-10 minutes total processing time
- **Transcription**: ~8 minutes (Whisper small.en model)
- **AI Analysis**: ~30-60 seconds (GPT-4o-mini)
- **Output**: 5-6 high-quality clip recommendations

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space (for model downloads)
- **CPU**: Multi-core recommended for faster processing
- **GPU**: Optional (CPU-only processing supported)

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python test_cli.py`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request