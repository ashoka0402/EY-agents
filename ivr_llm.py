from ivr_gemini import generate_text_gemini

def generate_ivr_message(explanation_list, language="English",
                         customer_name="Customer", vehicle_id="Your Vehicle"):

    template = open("prompt.txt").read()
    explanation_text = "\n".join(explanation_list)

    prompt = template.format(
        language=language,
        customer_name=customer_name,
        vehicle_id=vehicle_id,
        explanation=explanation_text
    )

    return generate_text_gemini(prompt)
