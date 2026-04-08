import streamlit as st
import pandas as pd
import time
from core.env import MediCoderEnv
from core.agent import get_action

# 1. Page Configuration & Branding
st.set_page_config(
    page_title="MediCoder AI | Revenue Integrity",
    page_icon="🏥",
    layout="wide"
)

# Professional CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .stButton>button { border-radius: 8px; font-weight: bold; height: 3em; }
    .stTextArea>div>div>textarea { background-color: #ffffff; border-radius: 10px; }
    
    [data-testid="column"]:nth-of-type(2) button {
        padding-top: 20px !important;
        padding-bottom: 20px !important;
        font-size: 16px !important; /* Slightly larger font */
        min-height: 4.5em !important; /* Increased height */
        background-color: #ffffff !important;
        border: 2px solid #3B82F6 !important; /* Highlighting it slightly */
    }

   
    .stButton>button { 
        border-radius: 8px; 
        font-weight: bold; 
        white-space: nowrap;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏥 MediCoder: Autonomous Medical Coding")
st.caption("Agentic RL Environment | Powered by Gemini 2.5 Flash")

# 2. Initialize Session State
if 'reward_history' not in st.session_state:
    st.session_state.reward_history = []
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'unsupported_memory' not in st.session_state:
    # This acts as our "Local Database" for the session
    st.session_state.unsupported_memory = []

# 3. Sidebar - Settings
with st.sidebar:
    st.header("⚙️ Configuration")
    max_steps = st.slider("Max Agent Retries", 1, 5, 3)
    
    st.divider()
    if st.button("Reset Session & Memory", use_container_width=True):
        st.session_state.reward_history = []
        st.session_state.logs = []
        st.session_state.unsupported_memory = [] # Clears the "Learned" invalid diseases
        st.rerun()

# 4. Main Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("🧬 Clinical Intelligence Input")
    
    c1, c2, c3, c4 = st.columns(4)
    if "selected_note" not in st.session_state: st.session_state.selected_note = ""

    if c1.button("Diabetes", width='stretch'):
        st.session_state.selected_note = "Patient presents with symptoms of type 2 diabetes mellitus."
    if c2.button("Hypertension", width='stretch'):
        st.session_state.selected_note = "Follow-up for primary hypertension. BP 140/90."
    if c3.button("Asthma", width='stretch'):
        st.session_state.selected_note = "Patient presenting with wheezing, diagnosed with bronchial asthma."
    if c4.button("Bronchitis", width='stretch'):
        st.session_state.selected_note = "Patient diagnosed with acute bronchitis and persistent cough."

    clinical_note = st.text_area("Physician Notes:", value=st.session_state.selected_note, height=150)
    run_btn = st.button("🚀 Run RL Coding Agent", type="primary", width='stretch')

# 5. Execution Logic with "Stateful Memory"
if run_btn and clinical_note:
    st.session_state.logs = []
    st.session_state.reward_history = []
    
    # Check if this note (or disease) is already in our "Unsupported" memory
    # We check if the first 20 characters match anything we've failed before
    note_id = clinical_note.lower()[:30] 
    
    if note_id in st.session_state.unsupported_memory:
        st.error("⚠️ System Memory: This clinical scenario was previously flagged as 'Unsupported' by current policy.")
        st.session_state.reward_history.append(-1.0)
        st.session_state.logs.append({
            "Try": 1, "ICD-10 Code": "N/A", "Score": -1.0, "Feedback": "Known Policy Violation (Memoized)"
        })
    else:
        env = MediCoderEnv(note=clinical_note)
        observation = env.reset()
        done = False
        step = 0
        
        with st.status("Agent analyzing...", expanded=True) as status:
            while not done and step < max_steps:
                step += 1
                
                # RL Agent Action
                full_response = get_action(observation)
                
                # Chain of Thought Parsing
                if "ACTION:" in full_response:
                    thought = full_response.split("ACTION:")[0].replace("THOUGHT:", "").strip()
                    action = full_response.split("ACTION:")[1].strip()
                else:
                    thought = "Analyzing documentation..."
                    action = full_response.strip()

                st.info(f"🧠 **Attempt {step} Thought:** {thought}")
                
                # Environment Feedback
                result = env.step([action])
                reward = result["reward"]
                done = result["done"]
                reason = result["info"]["reason"]

                # If the policy rejects it as unknown, add to Memory!
                if "not recognized" in reason.lower() or "violation" in reason.lower():
                    st.session_state.unsupported_memory.append(note_id)

                observation = f"Note: {clinical_note}. Previous Feedback: {reason}"
                
                st.session_state.reward_history.append(reward)
                st.session_state.logs.append({
                    "Try": step, "ICD-10 Code": action, "Score": reward, "Feedback": reason
                })
                time.sleep(0.6)
            status.update(label="Sequence Complete", state="complete")

# 6. Results Display
# 6. Results Display (Improved Layout)
with col2:
    st.subheader("📊 Policy Analytics")
    
    if st.session_state.reward_history:
        # Using a container to keep metrics together and visible
        with st.container():
            m1, m2 = st.columns(2)
            latest_reward = st.session_state.reward_history[-1]
            
            # Dynamic Status Label
            if latest_reward >= 1.0:
                status_label = "✅ VERIFIED"
                color = "normal"
            elif latest_reward < 0:
                status_label = "❌ PENALIZED"
                color = "inverse"
            else:
                status_label = "🔄 LEARNING"
                color = "off"

            m1.metric("Reward Score", f"{latest_reward:.2f}")
            m2.metric("System Status", status_label)

        # Optimization Graph
        st.markdown("### **Learning Curve**")
        st.line_chart(st.session_state.reward_history, height=250)
        
        # Audit Trail
        with st.expander("📝 View Detailed Audit Ledger"):
            st.dataframe(
                pd.DataFrame(st.session_state.logs), 
                use_container_width=True,
                hide_index=True
            )
            
        # Final Verification Message
        if latest_reward >= 1.0:
            st.success("**Policy Check Passed:** ICD-10 code is compliant.")
        else:
            st.error(f"**Policy Violation:** {st.session_state.logs[-1]['Feedback']}")
    else:
        st.info("Awaiting agent initiation...")
        st.info("Input a case to see real-time RL analytics.")

st.divider()
st.caption("🔒 MediCoder RL System | HIPAA-Compliant Architecture Simulation")