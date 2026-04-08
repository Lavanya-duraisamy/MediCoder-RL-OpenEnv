import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found! Check your .env file or Hugging Face Secrets.")

genai.configure(api_key=api_key)


model = genai.GenerativeModel('gemini-2.5-flash')
def get_medical_coding_action(observation):
    prompt = f"""
    STATE: {observation}
    TASK: Provide the most relevant ICD-10 code.
    FORMAT: Just the code (e.g., E11.9). No extra text.
    """
    try:
        response = model.generate_content(prompt)
        # Clean the response to ensure it's just the code string
        code = response.text.strip().split()[-1] 
        return [code] # Return as a LIST so core.policy stays happy
    except Exception as e:
        print(f"Agent Error: {e}")
        return ["ERROR"]