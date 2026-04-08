# core/env.py
from core.policy import verify_codes

class MediCoderEnv:
    def __init__(self, note):
        self.note = note
        self.state = {"note": note}
        self.done = False

    def reset(self):
        self.done = False
        return self.state["note"]

    def step(self, action):
        # Action is usually a list containing the ICD-10 string
        reward, status, reason = verify_codes(self.note, action)
        
        # If we hit +1.0, the "episode" is finished
        if reward >= 1.0:
            self.done = True
            
        return {
            "observation": self.state["note"],
            "reward": reward,
            "done": self.done,
            "info": {"status": status, "reason": reason}
        }