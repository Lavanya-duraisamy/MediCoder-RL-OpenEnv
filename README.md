---
title: MediCoder RL
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
MediCoder is a high-fidelity Reinforcement Learning (RL) environment designed to automate medical coding verification—a $30B problem in healthcare. It features an autonomous agent powered by GPT-4o-mini that extracts ICD-10 codes from clinical notes, which are then validated against a deterministic Policy Engine to ensure revenue integrity.

🛠️ Tech Stack
AI Agent: OpenAI GPT-4o-mini (Optimized for high-speed medical inference)

Environment: MediCoder-v1 (Fully OpenEnv Spec compliant)

Frameworks: FastAPI (Backend) & Streamlit (Frontend)

Logging: Standardized [START/STEP/END] JSON logging for automated grading.

Container: Dockerized for seamless deployment to Scaler/Meta cloud environments.

🏃 How to Run
Clone the Repo: git clone <your-repo-link>

Setup Environment: Create a .env file with:

Code snippet
OPENAI_API_KEY=your_actual_key
MODEL_NAME=gpt-4o-mini
Run Locally: Execute python main.py to launch both the API and the Dashboard.

Meta/Scaler Dry Run: Run python inference.py to see the structured RL logs.

📈 RL Reward Logic & Policy
Our environment uses a sparse-to-dense reward signal to train high-precision agents:

🎯 1.0 Reward (Accepted): Agent provided the correct ICD-10 prefix matching the clinical documentation.

⚠️ 0.1 Reward (Denied): Agent identified the correct disease but provided an incorrect or missing code.

❌ 0.0 Reward (Null): No clinical correlation found.
api/app.py: FastAPI implementation of the OpenEnv endpoints.
