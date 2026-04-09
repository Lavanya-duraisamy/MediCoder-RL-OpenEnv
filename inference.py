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
    # RULE: At least 3 tasks
    patient_cases = [
        {"id": "case-1", "note": "Type 2 diabetes with hyperglycemia."},
        {"id": "case-2", "note": "Acute chest pain and hypertension."},
        {"id": "case-3", "note": "Shortness of breath and chronic asthma."}
    ]

    for case in patient_cases:
        print(f"[START] task={case['id']} env=MediCoder-v1 model={MODEL_NAME}")
        sys.stdout.flush()
        
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
                    messages=[
                        {"role": "system", "content": "Output ONLY the ICD-10 code."},
                        {"role": "user", "content": f"Clinical Note: {current_obs}"}
                    ],
                    temperature=0.0
                )
                action = response.choices[0].message.content.strip()
                
                step_res = requests.post(f"{LOCAL_SERVER}/step", json={"action": [action]}).json()
                
                # RULE: Reward must be strictly between 0 and 1 (0.05 to 0.95)
                raw_reward = float(step_res.get("reward", 0.0))
                reward = max(0.05, min(0.95, raw_reward))
                
                done = bool(step_res.get("done", False))
                current_obs = step_res.get("obs")
                rewards.append(reward)
                
                print(f"[STEP] step={step_count} action={action} reward={reward:.2f} done={str(done).lower()} error=null")
                sys.stdout.flush()

            success = "true" if sum(rewards) > 0.1 else "false"
            print(f"[END] success={success} steps={step_count} rewards={','.join([f'{r:.2f}' for r in rewards])}")

        except Exception as e:
            print(f"[END] success=false steps={step_count} rewards=0.05 error={str(e)}")
        sys.stdout.flush()

if __name__ == "__main__":
    run_task()