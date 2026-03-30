"""
npc.py — Prescriber NPC behavior model.
Simulates doctor interactions and feedback.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import random

class PrescriberNPC:
    def __init__(self, name: str = "Dr. Smith", specialty: str = "Internal Medicine"):
        self.name = name
        self.specialty = specialty
        self.satisfaction = 0.5  # 0.0 to 1.0 (Short term)
        self.reputation = 0.5    # 0.0 to 1.0 (Long term)
        self.personality = random.choice(["cooperative", "defensive", "busy", "conservative"])
    
    def generate_response(self, agent_message: str, action_type: str, 
                         drug: str, patient_info: Optional[Dict] = None) -> str:
        """
        Produce a context-aware response from the prescriber.
        """
        msg = agent_message.lower()
        response = ""

        # Case 1: Agent is approving the prescription
        if action_type == "approve_prescription":
            if self.personality == "busy":
                response = f"[{self.name}]: Thanks. Proceeding with {drug}."
            else:
                response = f"[{self.name}]: Appreciate the quick review. We'll start {drug} now."
        
        # Case 2: Agent is modifying the prescription
        elif action_type == "modify_prescription":
            if "allergy" in msg or "allergic" in msg:
                response = f"[{self.name}]: Oh, I missed that allergy! Thank you. I agree with switching to {drug}."
                self.satisfaction += 0.1
            elif "resistance" in msg or "culture" in msg:
                response = f"[{self.name}]: Good catch on the lab results. I'm fine with {drug} if the sensitivity supports it."
                self.satisfaction += 0.05
            elif "renal" in msg or "egfr" in msg:
                response = f"[{self.name}]: True, forgot to dose-adjust for the kidney function. {drug} is safer."
            elif "de-escalat" in msg or "narrow" in msg:
                if self.personality == "conservative" and self.reputation < 0.7:
                    response = f"[{self.name}]: I'm hesitant to de-escalate. I've seen patients fail on narrow-spectrum therapy. I'd prefer sticking to the current plan."
                    self.satisfaction -= 0.05
                else:
                    response = f"[{self.name}]: Agreed. Stewardship is important. Let's switch to the narrower agent."
                    self.reputation += 0.02
            else:
                if self.personality == "defensive" or (self.reputation < 0.3):
                    response = f"[{self.name}]: Why {drug}? I thought my original choice was fine. Please explain the clinical benefit."
                    self.satisfaction -= 0.1
                else:
                    response = f"[{self.name}]: Understood. We'll update the order to {drug} as suggested."

        # Case 3: Agent is rejecting the prescription
        elif action_type == "reject_prescription":
            if not msg:
                response = f"[{self.name}]: You rejected the order without an explanation. I need to know why."
                self.satisfaction -= 0.2
            else:
                response = f"[{self.name}]: Rejection noted. I'll look for an alternative therapy."

        # Case 4: No action, just a message
        else:
            if "please check" in msg or "follow-up" in msg:
                response = f"[{self.name}]: Thanks for the heads up. I'll check the patient again."
            else:
                response = f"[{self.name}]: Received. Let me know if you have specific recommendations."

        self.satisfaction = max(0.0, min(1.0, self.satisfaction))
        self.reputation   = max(0.0, min(1.0, self.reputation))
        return response

def get_npc_for_task(task_id: str) -> PrescriberNPC:
    # Deterministic mapping for easier evaluation/debugging
    mapping = {
        "task_1": PrescriberNPC("Dr. Smith", "Internal Medicine"),
        "task_2": PrescriberNPC("Dr. Jones", "Urologist"),
        "task_3": PrescriberNPC("Dr. ICU-Lead", "Intensivist")
    }
    return mapping.get(task_id, PrescriberNPC())
