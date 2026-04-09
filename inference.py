import os
import sys
import requests
from openai import OpenAI

# 1. READ ENVIRONMENT VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "sk-placeholder"

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
LOCAL_SERVER = "http://0.0.0.0:7860"

def run_task():
    patient_cases = [
        {"id": "case-1", "note": "Diabetes case study."},
        {"id": "case-2", "note": "Cardiac emergency note."},
        {"id": "case-3", "note": "Respiratory distress note."}
    ]

    for case in patient_cases:
        # LOGGING START - Use flush=True
        print(f"[START] task={case['id']} env=MediCoder-v1 model={MODEL_NAME}", flush=True)
        
        rewards = []
        step_count = 0
        done = False
        
        try:
            res = requests.post(f"{LOCAL_SERVER}/reset", json={"note": case["note"]}).json()
            current_obs = res.get("obs")
            
            while not done and step_count < 3:
                step_count += 1
                
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": f"Medical code for: {current_obs}"}],
                    temperature=0.0
                )
                action = response.choices[0].message.content.strip().replace(" ", "_")
                
                step_res = requests.post(f"{LOCAL_SERVER}/step", json={"action": [action]}).json()
                
                # Use the fractional reward from our server
                reward = float(step_res.get("reward", 0.05))
                done = bool(step_res.get("done", False))
                current_obs = step_res.get("obs")
                rewards.append(reward)
                
                # LOGGING STEP - Use flush=True
                print(f"[STEP] step={step_count} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

            # LOGGING END - Use flush=True
            # Important: Ensure success is lowercase boolean
            success = "true" if any(r > 0.1 for r in rewards) else "false"
            rewards_str = ",".join([f"{r:.2f}" for r in rewards])
            print(f"[END] success={success} steps={step_count} rewards={rewards_str}", flush=True)

        except Exception as e:
            # Fallback END line so the parser doesn't fail on crash
            print(f"[END] success=false steps={step_count} rewards=0.05 error={str(e)}", flush=True)

if __name__ == "__main__":
    run_task()