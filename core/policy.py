DISEASE_POLICIES = {
    "diabetes": {"name": "Type 2 Diabetes", "keywords": ["diabetes", "sugar", "glucose"], "required_prefix": "E11"},
    "hypertension": {"name": "Hypertension", "keywords": ["hypertension", "blood pressure", "bp"], "required_prefix": "I10"},
    "asthma": {"name": "Asthma", "keywords": ["asthma", "wheezing", "bronchial"], "required_prefix": "J45"}
}

def verify_codes(note: str, proposed_codes: list, invalid_history: list = None):
    note_lower = note.lower()
    
    # Check if the current note matches something we already flagged as invalid
    if invalid_history:
        for invalid_term in invalid_history:
            if invalid_term in note_lower:
                return -1.0, "denied", f"History Match: '{invalid_term}' confirmed unsupported."

    for disease, policy in DISEASE_POLICIES.items():
        if any(k in note_lower for k in policy["keywords"]):
            # This line will now work because proposed_codes is a list
            match = any(c.strip().upper().startswith(policy["required_prefix"]) for c in proposed_codes)
            if match:
                return 1.0, "accepted", f"Success: {policy['name']} policy satisfied."
            return -0.5, "denied", f"Prefix Mismatch for {policy['name']}."
                
    return -1.0, "denied", "Policy Violation: Disease not recognized."