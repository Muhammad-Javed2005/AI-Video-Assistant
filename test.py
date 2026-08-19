import os
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import generate_title, summarize

# Test YouTube Video URL
YOUTUBE_URL = "https://youtu.be/FobiKSh8IFE?si=wuBGQm--JxPq4hQy"


def main():
    print("=== AI Video Assistant - Full Pipeline Test ===")

    # Step 1: Download & Chunking
    print("\n[1/4] 📥 Downloading & Chunking Audio...")
    chunks = process_input(YOUTUBE_URL)
    print(f"[✔] Total {len(chunks)} chunk(s) ready for transcription.")

    # Step 2: Transcription (Faster-Whisper Auto-Translate to English)
    print("\n[2/4] 🎙️ Transcribing & Translating Audio to English...")
    transcript = transcribe_all(chunks)
    print(f"[✔] Transcription completed. (Length: {len(transcript)} characters)")

    # Step 3: Generate Title (Mistral AI)
    print("\n[3/4] 🏷️ Generating Title using Mistral AI...")
    title = generate_title(transcript)
    print(f"[✔] Title Generated!")

    # Step 4: Summarize (Mistral AI Map-Reduce)
    print("\n[4/4] 📝 Generating Summary using Mistral AI...")
    summary = summarize(transcript)
    print(f"[✔] Summary Generated!")


    # Step 5: Print Final Results in Terminal
    print("\n\n" + "═" * 60)
    print("🎯 FINAL OUTPUTS")
    print("═" * 60)

    print("\n📌 TITLE:")
    print("-" * 20)
    print(title)

    print("\n📝 SUMMARY:")
    print("-" * 20)
    print(summary)

    print("\n📜 FULL ENGLISH TRANSCRIPT:")
    print("-" * 20)
    print(transcript)

    print("\n" + "═" * 60)
    print("🎉 All tasks completed successfully!")


if __name__ == "__main__":
    main()