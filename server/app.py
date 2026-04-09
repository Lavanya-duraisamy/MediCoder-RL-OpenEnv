import os
import sys
import uvicorn
import subprocess
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

# 1. PATH FIX: Ensure 'core' is discoverable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. DEFENSIVE IMPORTS
try:
    from core.env import MediCoderEnv
    # We keep the policy checker, but we don't need the internal agent here anymore
    from core.policy import verify_codes
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")

app = FastAPI(title="Medi-Coder RL Backend")

# 3. STATE MANAGEMENT
class PatientState:
    def __init__(self):
        self.current_observation = "Initial state: Patient presenting with symptoms."
        self.step_count = 0

state = PatientState()

# 4. REQUEST MODELS
class ResetRequest(BaseModel):
    note: Optional[str] = "Default Clinical Note"

class StepAction(BaseModel):
    action: Optional[List[str]] = None
    codes: Optional[List[str]] = None

# 5. ENDPOINTS
@app.get("/")
async def health_check():
    return {
        "status": "online", 
        "port": 7860, 
        "system": "Medi-Coder RL Engine",
        "ready": True
    }

@app.post("/reset")
async def reset(data: Optional[ResetRequest] = None):
    try:
        note_to_use = data.note if (data and data.note) else "Default Clinical Note"
        state.current_observation = str(note_to_use)
        state.step_count = 0
        
        return {
            "obs": state.current_observation, 
            "info": {"status": "initialized"}
        }
    except Exception as e:
        # Standardized to 'obs'
        return {"obs": "Error during reset", "info": {"error": str(e)}}

@app.post("/step")
async def step(data: Optional[StepAction] = None):
    try:
        state.step_count += 1
        
        # Take the action sent by inference.py (the LLM's output)
        proposed = []
        if data:
            if data.action:
                proposed = data.action
            elif data.codes:
                proposed = data.codes
        
        # ENVIRONMENT RULE: If inference.py fails to provide a code, we give 0 reward
        if not proposed:
            return {
                "obs": str(state.current_observation),
                "reward": 0.0,
                "done": False,
                "info": {"status": "error", "reason": "No code provided by agent"}
            }
        
        # Verify the LLM's proposed codes against our local policy
        reward, status, reason = verify_codes(state.current_observation, proposed)
        
        is_done = bool((status == "accepted") or (state.step_count >= 3))
        
        return {
            "obs": str(state.current_observation),
            "reward": float(reward),
            "done": is_done,
            "info": {
                "status": str(status),
                "reason": str(reason),
                "step_count": int(state.step_count),
                "proposed_codes": proposed
            }
        }
    except Exception as e:
        return {
            "obs": str(state.current_observation),
            "reward": -1.0,
            "done": True,
            "info": {"error": str(e)}
        }

def main():
    # Start Streamlit in the background
    script_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.py")
    if os.path.exists(script_path):
        print("Starting Frontend Dashboard...")
        subprocess.Popen([
            "streamlit", "run", script_path, 
            "--server.port=7861", 
            "--server.address=0.0.0.0",
            "--server.headless=true"
        ])
    
    print("CRITICAL: Starting Uvicorn on 7860...")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")

if __name__ == "__main__":
    main()