import os
import subprocess
import sys

# This "Entry Point" allows Hugging Face to launch your Streamlit UI
if __name__ == "__main__":
    # Path to your main Streamlit file
    script_path = os.path.join("frontend", "index.py")
    
    # Run Streamlit as a subprocess
    subprocess.run([
        "streamlit", 
        "run", 
        script_path, 
        "--server.port=7860", 
        "--server.address=0.0.0.0"
    ])