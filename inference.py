import os
import sys
import json
import time
from openai import OpenAI
from core.env import MediCoderEnv 

# Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=API_BASE_URL
)

def log_structured(status, data=None):
    """Logs in [START], [STEP], [END] format for Meta's grader."""
    log_entry = {"status": status, "timestamp": time.time()}
    if data:
        log_entry.update(data)
    print(f"[{status.upper()}] {json.dumps(log_entry)}")
    sys.stdout.flush()

def run_inference():
    log_structured("start", {"model": MODEL_NAME, "env": "MediCoder-v1"})
    
    # Sample note for the grader to evaluate
    sample_note = "Patient presents with type 2 diabetes and high blood pressure."
    env = MediCoderEnv(note=sample_note)
    
    observation = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    try:
        while not done and step_count < 3:
            # LLM selects the ICD-10 code
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a medical coder. Output ONLY the ICD-10 code."},
                    {"role": "user", "content": f"Clinical Note: {observation}"}
                ],
                temperature=0.0
            )
            
            action = response.choices[0].message.content.strip()
            
            # Environment Step
            result = env.step([action])
            
            reward = result["reward"]
            done = result["done"]
            observation = result["observation"]
            
            total_reward += reward
            step_count += 1
            
            log_structured("step", {
                "step": step_count,
                "action": action,
                "reward": reward,
                "total_reward": total_reward
            })

        log_structured("end", {"final_reward": total_reward, "steps": step_count})
        
    except Exception as e:
        log_structured("error", {"message": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    run_inference()