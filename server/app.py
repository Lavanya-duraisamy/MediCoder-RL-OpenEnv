import sys
import os
import uvicorn
import subprocess
from typing import List, Optional
from fastapi import FastAPI, Request
from pydantic import BaseModel

# 1. CRITICAL: Add root path BEFORE importing from 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.env import MediCoderEnv
    from core.agent import get_medical_coding_action 
    from core.policy import verify_codes
except ImportError as e:
    print(f"Import Error: {e}")

app = FastAPI(title="Medi-Coder RL Backend")

class PatientState:
    def __init__(self):
        self.current_observation = "Initial state"
        self.step_count = 0

state = PatientState()

class ResetRequest(BaseModel):
    note: Optional[str] = "Default Clinical Note"

class StepAction(BaseModel):
    action: Optional[List[str]] = None
    codes: Optional[List[str]] = None

@app.get("/")
async def health_check():
    return {"status": "online", "port": 7860, "system": "Medi-Coder RL Engine"}

@app.post("/reset")
async def reset(data: Optional[ResetRequest] = None):
    try:
        note_to_use = data.note if (data and data.note) else "Default Clinical Note"
        state.current_observation = note_to_use
        state.step_count = 0
        return {
            "observation": str(state.current_observation), 
            "info": {"status": "initialized"}
        }
    except Exception as e:
        return {"observation": "Error", "info": {"error": str(e)}}

@app.post("/step")
async def step(data: Optional[StepAction] = None):
    try:
        state.step_count += 1
        
        # Handle the validator sending empty bodies
        proposed = []
        if data:
            proposed = data.action if data.action else (data.codes if data.codes else [])
        
        # If no codes provided, ask the agent
        if not proposed:
            agent_result = get_medical_coding_action(state.current_observation)
            # Ensure agent_result is a list
            proposed = agent_result if isinstance(agent_result, list) else [str(agent_result)]
        
        reward, status, reason = verify_codes(state.current_observation, proposed)
        
        # Force strict types for the JSON response
        done = bool((status == "accepted") or (state.step_count >= 3))
        
        return {
            "observation": str(state.current_observation),
            "reward": float(reward),
            "done": done,
            "info": {
                "status": str(status),
                "reason": str(reason),
                "step_count": int(state.step_count),
                "proposed_codes": proposed
            }
        }
    except Exception as e:
        # Fallback to prevent inference.py from crashing
        return {
            "observation": "Error",
            "reward": -1.0,
            "done": True,
            "info": {"error": str(e)}
        }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    # Start Streamlit in background on secondary port
    script_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.py")
    if os.path.exists(script_path):
        subprocess.Popen([
            "streamlit", "run", script_path, 
            "--server.port=7861", 
            "--server.address=0.0.0.0"
        ])
    
    main()