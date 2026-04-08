import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found! Check your .env file or Hugging Face Secrets.")

genai.configure(api_key=api_key)


model = genai.GenerativeModel('gemini-2.5-flash')
def get_action(observation):
    prompt = f"""
    STATE: {observation}
    
    INSTRUCTIONS:
    - Act as an Autonomous RL Coding Agent.
    - If a penalty was previously received, justify the policy correction.
    - If successful, provide a high-confidence summary.
    - Limit response to 15 words.

    FORMAT:
    THOUGHT: [Policy-driven reasoning]
    ACTION: [ICD-10 Code]
    """
   
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"THOUGHT: API Error - {str(e)}\nACTION: ERROR"