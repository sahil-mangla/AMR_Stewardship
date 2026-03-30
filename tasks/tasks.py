"""
tasks.py — Clinical Task Definitions and Deterministic Graders.
Compliant with OpenEnv specification.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from env.models import ActionType, PrescriptionStatus


@dataclass
class TaskDefinition:
    task_id: str
    name: str
    difficulty: str
    description: str

ALL_TASKS = {
    "task_1": TaskDefinition(
        "task_1", "Allergy Safety Check", "easy",
        "Patient P-001 (68F) with documented penicillin allergy prescribed amoxicillin."
    ),
    "task_2": TaskDefinition(
        "task_2", "Resistance-Guided De-escalation", "medium",
        "E. coli UTI (P-002) prescribed ciprofloxacin. Antibiogram shows high resistance. Lab confirms resistance."
    ),
    "task_3": TaskDefinition(
        "task_3", "ICU Multi-Drug Stewardship", "hard",
        "Complex ICU patient (P-003) with MRSA bacteremia, severe renal impairment (eGFR=22), and multi-drug allergies."
    )
}

def list_tasks() -> List[str]:
    return list(ALL_TASKS.keys())

# ── Programmatic Graders ──────────────────────────────────────────────────

def _grade_task1(episode_history: List[Dict]) -> float:
    """
    Allergy Check Grader.
    Success: Modify to non-penicillin OR Reject with allergy reason.
    Failure: Approve penicillin.
    """
    for entry in episode_history:
        action = entry.get("action", {})
        atype = action.get("action_type")
        params = action.get("parameters", {})
        
        if atype == ActionType.APPROVE_PRESCRIPTION:
            return 0.0 # Critical safety failure
        
        if atype == ActionType.MODIFY_PRESCRIPTION:
            drug = params.get("drug", "").lower()
            if drug and drug not in ["penicillin", "amoxicillin", "ampicillin", "piperacillin-tazobactam"]:
                return 1.0
        
        if atype == ActionType.REJECT_PRESCRIPTION:
            reason = params.get("reason", "").lower()
            if "allergy" in reason:
                return 1.0
                
    return 0.0

def _grade_task2(episode_history: List[Dict]) -> float:
    """
    De-escalation Grader.
    Success: Modify ciprofloxacin to nitrofurantoin (first-line narrow-spectrum).
    Partial: Correct decision but wrong tool flow.
    """
    score = 0.0
    de_escalated = False
    
    # Check if tools were used correctly (Clinical rationalism)
    p_data = l_data = a_data = False
    
    for entry in episode_history:
        action = entry.get("action", {})
        atype = action.get("action_type")
        
        if atype == ActionType.GET_PATIENT_DATA.value: p_data = True
        if atype == ActionType.GET_LAB_RESULTS.value: l_data = True
        if atype == ActionType.CHECK_ANTIBIOGRAM.value: a_data = True
        
        if atype == ActionType.MODIFY_PRESCRIPTION.value:
            drug = action.get("parameters", {}).get("drug", "").lower()
            if drug == "nitrofurantoin":
                de_escalated = True
    
    if de_escalated:
        score = 0.7
        if p_data and l_data and a_data: score = 1.0 # Bonus for clinical evidentiary trail
        
    return score

def _grade_task3(episode_history: List[Dict]) -> float:
    """
    Complex ICU Grader.
    Success: 
      - Adjust Vancomycin (Modify to lower dose/freq)
      - Justify Meropenem (Message explaining ICU benefit)
    """
    vanco_adjusted = False
    mero_justified = False
    
    for entry in episode_history:
        action = entry.get("action", {})
        atype = action.get("action_type")
        params = action.get("parameters", {})
        
        if atype == ActionType.MODIFY_PRESCRIPTION.value:
            drug = params.get("drug", "").lower()
            if "vancomycin" in drug:
                # Check for renal adjustment (lower dose or frequency)
                freq = params.get("frequency", "").lower()
                if "q24" in freq or "q48" in freq or "q72" in freq:
                    vanco_adjusted = True
        
        if atype == ActionType.SEND_MESSAGE.value:
            msg = params.get("message", "").lower()
            if "restricted" in msg or "meropenem" in msg or "icu" in msg:
                mero_justified = True
                
    score = 0.0
    if vanco_adjusted: score += 0.5
    if mero_justified: score += 0.5
    
    return score

# ── Main Entry Point for Graders ──────────────────────────────────────────

def run_grader(task_id: str, episode_history: List[Dict]) -> float:
    """Dispatches the appropriate grader for a given task ID."""
    if task_id == "task_1":
        return _grade_task1(episode_history)
    elif task_id == "task_2":
        return _grade_task2(episode_history)
    elif task_id == "task_3":
        return _grade_task3(episode_history)
    return 0.0
