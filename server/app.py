import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import subprocess

# Now that the path is set, these imports will work
from core.env import MediCoderEnv
from core.agent import get_medical_coding_action 
from core.policy import verify_codes

app = FastAPI(title="Medi-Coder RL Backend")

class PatientState:
    def __init__(self):
        self.current_observation = ""
        self.step_count = 0

state = PatientState()

class ResetRequest(BaseModel):
    note: str

class StepAction(BaseModel):
    # Standardizing 'action' for OpenEnv and 'codes' for your UI
    action: Optional[List[str]] = None
    codes: Optional[List[str]] = None

@app.get("/")
async def health_check():
    return {"status": "online", "port": 7860, "system": "Medi-Coder RL Engine"}

@app.post("/reset")
async def reset(data: ResetRequest):
    state.current_observation = data.note
    state.step_count = 0
    return {"observation": state.current_observation, "info": {"status": "initialized"}}

@app.post("/step")
async def step(data: StepAction):
    state.step_count += 1
    
    # Check for 'action' (validator standard) or 'codes' (UI standard)
    proposed = data.action if data.action else (data.codes if data.codes else [])
    
    if not proposed:
        proposed = [get_medical_coding_action(state.current_observation)]
    
    reward, status, reason = verify_codes(state.current_observation, proposed)
    done = (status == "accepted") or (state.step_count >= 3)
    
    return {
        "observation": state.current_observation,
        "reward": reward,
        "done": done,
        "info": {
            "status": status,
            "reason": reason,
            "step_count": state.step_count,
            "proposed_codes": proposed
        }
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    # Correct path to find the frontend from the server folder
    script_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.py")
    if os.path.exists(script_path):
        subprocess.Popen([
            "streamlit", "run", script_path, 
            "--server.port=7861", 
            "--server.address=0.0.0.0"
        ])
    
    main()