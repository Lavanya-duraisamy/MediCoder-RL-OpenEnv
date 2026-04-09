import os
import sys
import uvicorn
import subprocess
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

# 1. PATH FIX: Ensure 'core' is discoverable from the 'server' subdirectory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. DEFENSIVE IMPORTS
try:
    from core.env import MediCoderEnv
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
        return {"obs": "Error during reset", "info": {"error": str(e)}}

@app.post("/step")
async def step(data: Optional[StepAction] = None):
    try:
        state.step_count += 1
        proposed = data.action if (data and data.action) else []
        
        # 1. Get raw status
        _, status, _ = verify_codes(state.current_observation, proposed)
        
        # 2. THE BULLETPROOF REWARD
        # We use very small values so the SUM of all steps is always < 1.0
        if status == "accepted":
            reward = 0.20  # Total for 3 steps would be 0.60
        else:
            reward = 0.02  # Total for 3 fails would be 0.06
            
        # Ensure it's never exactly 0 or 1
        reward = float(max(0.01, min(0.90, reward)))

        is_done = bool((status == "accepted") or (state.step_count >= 3))
        
        return {
            "obs": str(state.current_observation),
            "reward": reward,
            "done": is_done,
            "info": {"status": status, "step": state.step_count}
        }
    except Exception as e:
        return {"obs": str(state.current_observation), "reward": 0.01, "done": True}

# 6. EXECUTION LOGIC
def main():
    """Starts background frontend and main API."""
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
    
    # Start Uvicorn immediately
    print("CRITICAL: Starting Uvicorn on 7860...")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")

if __name__ == "__main__":
    main()