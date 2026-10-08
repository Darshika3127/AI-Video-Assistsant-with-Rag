# AI Video Assistant

An AI-powered meeting and video assistant that turns audio or YouTube content into a structured transcript, summary, action items, decisions, and a searchable Q&A interface.

This project combines local speech-to-text, summarization, extraction, and retrieval-augmented generation (RAG) to help users quickly understand long recordings without reading every minute manually.

## Features

- Download audio from YouTube URLs
- Convert local audio/video files into a processable WAV format
- Split long audio into chunks for reliable transcription
- Transcribe speech using local Whisper or Sarvam AI for Hinglish input
- Generate:
  - meeting title
  - summary
  - action items
  - key decisions
  - open questions
- Build a local RAG workflow over the transcript for follow-up questions
- Provide a Streamlit-based UI for interactive use

## Tech Stack

- Python 3.10+
- Streamlit
- Whisper (local transcription)
- Sarvam AI STT translation API (optional for Hinglish)
- LangChain + Ollama
- ChromaDB for vector storage
- SentenceTransformers embeddings
- PyDub, yt-dlp, ffmpeg

## Project Structure

```text
AI-Video-Assistant--main/
├── app.py                 # Streamlit UI entry point
├── main.py                # CLI pipeline runner
├── Requirements.txt       # Python dependencies
├── .env                   # Environment variables
├── downloades/            # Downloaded/generated audio files
├── core/
│   ├── extractor.py       # Extract action items / decisions / questions
│   ├── llm_client.py      # Ollama LLM client with retry handling
│   ├── rag_engine.py      # RAG setup and Q&A logic
│   ├── summarizer.py      # Title + meeting summary generation
│   ├── transcriber.py     # Audio transcription pipeline
│   └── vector_store.py    # Vector DB interactions
├── utils/
│   └── audio_processor.py # YouTube/local audio handling
├── vector_db/             # Vector database storage
└── README.md
```

## Prerequisites

Before running the app, install:

- Python 3.10 or newer
- FFmpeg
- Ollama
- A local model compatible with the app configuration, such as `qwen3:8b`

Install the model:

```bash
ollama pull qwen3:8b
```

## Installation

```bash
cd AI-Video-Assistant--main
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r Requirements.txt
```

## Environment Variables

Create a `.env` file in the project root if needed:

```env
SARVAM_API_KEY=your_sarvam_api_key_here
WHISPER_MODEL=medium
```

Notes:
- `SARVAM_API_KEY` is optional unless you plan to use the Hinglish transcription flow.
- The app expects Ollama to be running locally at `http://localhost:11434`.

## Usage

### Streamlit app

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit in your browser and use the interface to upload or link a video/audio source.

### CLI mode

```bash
python main.py
```

Then enter:
- a YouTube URL or local file path
- the language (`english` or `hinglish`)

The pipeline will:
1. download/process the input,
2. transcribe it,
3. generate structured notes,
4. enable chat-based Q&A over the transcript.

## Example Workflow

- Paste a YouTube meeting link
- Wait for audio download and chunking
- Get transcript + AI summary
- Read generated action items and decisions
- Ask questions like: "What were the key blockers?" or "What is the next step?"

## Notes

- The app is designed for local-first AI workflows and may require a strong GPU or sufficient CPU resources depending on the Whisper and embedding models used.
- For best results, make sure Ollama is running and the selected model is available before starting the app.

## License

This project does not include a license file. If you are publishing or distributing it externally, add an appropriate license before use.
