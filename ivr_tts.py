# ivr_tts.py
import os
from elevenlabs import ElevenLabs, VoiceSettings
from dotenv import load_dotenv
load_dotenv()

def generate_voice_audio(
    text: str,
    output_path: str = "ivr_message.wav",
    voice_id: str = "EXAVITQu4vr4xnSDxMaL",  # Bella's voice ID
    stability: float = 0.45,
    similarity_boost: float = 0.85
):
    """
    Generate high-quality TTS audio using ElevenLabs.
    
    Args:
        text (str): The text to convert into speech.
        output_path (str): Where to save the .wav file.
        voice_id (str): ElevenLabs voice ID (default is Bella).
        stability (float): Voice stability (0–1).
        similarity_boost (float): Naturalness tuning.
    """

    # Initialize ElevenLabs client
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set")
    
    client = ElevenLabs(api_key=api_key)

    # Generate audio
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",   # Supports Hindi, Tamil, Marathi, Bengali, English
        voice_settings=VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost
        )
    )

    # Save output file
    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    print(f"Audio saved to: {output_path}")
    return output_path


# Standalone test
if __name__ == "__main__":
    test_text = "Hello! Your service appointment is recommended. Press 1 to confirm."
    try:
        file_path = generate_voice_audio(test_text, "test_voice.wav")
        print("Success! Saved:", file_path)
    except Exception as e:
        print(f"Error generating audio: {e}")