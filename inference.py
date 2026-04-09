import os
import sys
import requests
from openai import OpenAI

# 1. READ ENVIRONMENT VARIABLES (As injected by Scaler/Meta)
# The validator will overwrite these; your defaults are only for local safety
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
# They explicitly asked for API_KEY or HF_TOKEN
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

if not API_KEY:
    raise ValueError("Missing API_KEY/HF_TOKEN from validator environment")

# 2. INITIALIZE CLIENT (Must use their Proxy)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# Your server's internal address
LOCAL_SERVER = "http://0.0.0.0:7860"

def run_task():
    print(f"[START] task=medical-coding env=MediCoder-v1 model={MODEL_NAME}")
    sys.stdout.flush()
    
    rewards = []
    step_count = 0
    done = False
    
    try:
        # Step 1: Reset your local environment
        reset_res = requests.post(f"{LOCAL_SERVER}/reset").json()
        current_obs = reset_res.get("obs")
        
        while not done and step_count < 3:
            step_count += 1
            
            # Step 2: MAKE THE PROXY API CALL (This is what they are looking for)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a medical coder. Output ONLY the ICD-10 code."},
                    {"role": "user", "content": f"Clinical Note: {current_obs}"}
                ],
                temperature=0.0
            )
            action = response.choices[0].message.content.strip()
            
            # Step 3: Send that action to YOUR server to get reward/done status
            step_res = requests.post(f"{LOCAL_SERVER}/step", json={"action": [action]}).json()
            
            reward = float(step_res.get("reward", 0.0))
            done = bool(step_res.get("done", False))
            current_obs = step_res.get("obs")
            rewards.append(reward)
            
            print(f"[STEP] step={step_count} action={action} reward={reward:.2f} done={str(done).lower()} error=null")
            sys.stdout.flush()

        success = "true" if sum(rewards) > 0 else "false"
        print(f"[END] success={success} steps={step_count} rewards={','.join([f'{r:.2f}' for r in rewards])}")

    except Exception as e:
        print(f"[END] success=false steps={step_count} rewards=0.00 error={str(e)}")
    sys.stdout.flush()

if __name__ == "__main__":
    run_task()