import os
from faster_whisper import WhisperModel

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
_whisper_model = None


def load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Faster-Whisper ({WHISPER_MODEL_NAME}) on CPU...")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_NAME, device="cpu", compute_type="int8", cpu_threads=4
        )
        print("Faster-Whisper model loaded successfully.")
    return _whisper_model


def transcribe_chunk_whisper(chunk_path: str, force_english: bool = True) -> str:
    model = load_whisper_model()
    task_type = "translate" if force_english else "transcribe"

    segments, info = model.transcribe(
        chunk_path,
        task=task_type,
        beam_size=5,
        vad_filter=True,  # Blank/silent noise auto-skip
    )

    text_list = [segment.text for segment in segments]
    return " ".join(text_list).strip()


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = []
    print("\n--- Starting Audio Transcription [Faster-Whisper] ---")

    for i, chunk in enumerate(chunks):
        print(f"[+] Processing Chunk {i + 1}/{len(chunks)} ({chunk})...")
        text = transcribe_chunk_whisper(chunk, force_english=True)
        print(f"[✔] Chunk {i + 1} Completed!")
        full_transcript.append(text)

    print("\n[🎉] All Chunks Transcribed Successfully!")
    return "\n\n".join(full_transcript)