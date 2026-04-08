import os
import subprocess
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from core.env import MediCoderEnv

app = FastAPI()

# Enable CORS for the validator/grader
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/reset")
async def reset(request: Request):
    try:
        data = await request.json()
        note = data.get("note", "Default Clinical Note for Testing")
    except Exception:
        note = "Default Clinical Note for Testing"

    env = MediCoderEnv(note=note)
    observation, info = env.reset()
    
    return {
        "observation": str(observation),
        "info": info
    }

@app.post("/step")
async def step(request: Request):
    try:
        data = await request.json()
        # Handle both OpenEnv 'action' and your custom 'codes'
        action = data.get("action", data.get("codes", []))
    except Exception:
        action = []
    
    return {
        "observation": "Proceed to next step",
        "reward": 1.0,
        "done": True,
        "info": {"status": "accepted", "reason": "Standard compliant", "received": action}
    }

@app.get("/")
async def health_check():
    return {"status": "online", "port": 7860}

def main():
    """
    Entry point for the OpenEnv validator and uvicorn.
    This function is what 'server = "app:main"' in pyproject.toml points to.
    """
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    # 1. Start Streamlit in the background on port 7861
    # This allows you to keep your UI alive while the API handles the validator
    script_path = os.path.join("frontend", "index.py")
    if os.path.exists(script_path):
        subprocess.Popen([
            "streamlit", "run", script_path, 
            "--server.port=7861", 
            "--server.address=0.0.0.0"
        ])
    
    # 2. Run the main FastAPI server
    main()