import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyBNMdVooGveZmGDt_EEgAyRE3-ZE0M3Aqg")

def generate_text_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text
