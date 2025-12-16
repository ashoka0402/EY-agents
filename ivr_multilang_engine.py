from ivr_llm import generate_ivr_message
from ivr_tts import generate_voice_audio

def create_multilang_ivr(
    explanation_list,
    language="English",
    customer_name="Customer",
    vehicle_id="Vehicle-001",
    output_audio="ivr_output.wav"
):
    # Step 1: Generate text in requested language
    ivr_text = generate_ivr_message(
        explanation_list=explanation_list,
        language=language,
        customer_name=customer_name,
        vehicle_id=vehicle_id
    )

    print("\nGenerated IVR Text:\n", ivr_text)

    # Step 2: Convert to speech (female voice)
    audio_file = generate_voice_audio(
        text=ivr_text,
        output_path=output_audio,
        voice_id="EXAVITQu4vr4xnSDxMaL"  # Bella's voice ID (changed from voice_name)
    )

    print("\nAudio saved at:", audio_file)
    return ivr_text, audio_file


# DEMO
if __name__ == "__main__":
    explanation_list = [
        "Engine temperature is higher than usual.",
        "Brake components show abnormal heat patterns.",
        "Driving style appears aggressive recently.",
        "Immediate service recommended."
    ]

    # Try different languages
    for lang in ["English", "Hindi"]:
        print(f"\n--- Generating IVR in {lang} ---")
        try:
            create_multilang_ivr(
                explanation_list=explanation_list,
                language=lang,
                customer_name="Rahul",
                vehicle_id="MH12AB1234",
                output_audio=f"ivr_{lang}.wav"
            )
        except Exception as e:
            print(f"Error generating IVR for {lang}: {e}")