from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
# CHANGE THIS LINE:
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
    codes: Optional[List[str]] = None

@app.get("/")
async def health_check():
    return {"status": "online", "system": "Medi-Coder RL Engine"}

@app.post("/reset")
async def reset(data: ResetRequest):
    state.current_observation = data.note
    state.step_count = 0
    return {"observation": state.current_observation}

@app.post("/step")
async def step(action: StepAction):
    state.step_count += 1
    
    # CHANGE THIS LINE:
    proposed = action.codes if action.codes else [get_medical_coding_action(state.current_observation)]
    
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)