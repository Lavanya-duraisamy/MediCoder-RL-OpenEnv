import subprocess
import time
import os
import sys

def start_backend():
    # Starts the FastAPI app on port 8000
    cmd = [sys.executable, "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
    return subprocess.Popen(cmd)

def start_frontend():
    # Starts the Streamlit dashboard on port 8501
    frontend_path = os.path.join("frontend", "index.py")
    cmd = [sys.executable, "-m", "streamlit", "run", frontend_path, "--server.port", "8501", "--server.address", "0.0.0.0"]
    return subprocess.Popen(cmd)

if __name__ == "__main__":
    print("🏥 Starting Medi-Coder RL Local Environment...")
    
    # 1. Start FastAPI
    backend = start_backend()
    print("⏳ Waiting for Backend to initialize...")
    time.sleep(5) 
    
    # 2. Start Streamlit
    frontend = start_frontend()
    
    print("\n✅ System Online!")
    print("Dashboard: http://localhost:8501")
    print("API Docs:  http://localhost:8000/docs\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        backend.terminate()
        frontend.terminate()