"""
tasks.py — Three tasks with deterministic graders (easy → medium → hard).
"""
from __future__ import annotations
import time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from env.models import PrescriptionRequest, PrescriptionStatus


@dataclass
class Task:
    task_id: str
    description: str
    difficulty: str
    prescriptions: List[PrescriptionRequest]
    task_context: Dict[str, Any]
    grader: Callable


def _rx(id,pid,pres,drug,dose,route,freq,dur,ind):
    return PrescriptionRequest(
        id=id, patient_id=pid, prescriber_id=pres, drug=drug,
        dose_mg=dose, route=route, frequency=freq, duration_days=dur,
        indication=ind, status=PrescriptionStatus.PENDING, created_at=time.time()
    )

# ── TASK 1 — EASY ────────────────────────────────────────────────────────────

def _grade_task1(decisions_made, messages_sent):
    if not decisions_made: return 0.0
    last = decisions_made[-1]
    atype = last.get("action_type","")
    drug  = last.get("drug","").lower()
    has_msg = len(messages_sent) > 0
    PENICILLIN = {"penicillin","amoxicillin","ampicillin","amoxicillin-clavulanate"}
    SAFE       = {"azithromycin","doxycycline","levofloxacin","ceftriaxone"}
    if atype == "approve_prescription": return 0.0
    score = 0.0
    if atype == "modify_prescription" and drug in SAFE:
        score = 0.75
    elif atype == "reject_prescription":
        reason = last.get("reason","").lower()
        score = 0.65 if ("allerg" in reason or "penicillin" in reason) else 0.25
    if has_msg: score = min(1.0, score + 0.25)
    return round(score, 4)

TASK_1 = Task(
    task_id="task_1", difficulty="easy",
    description=(
        "EASY — Allergy Safety Check\n\n"
        "Patient P001 (68F) has a documented PENICILLIN ALLERGY.\n"
        "Prescriber ordered AMOXICILLIN 500mg PO q8h × 7 days for community-acquired pneumonia.\n\n"
        "Your job:\n"
        "1. Identify the allergy conflict.\n"
        "2. Reject or modify the prescription.\n"
        "3. Suggest a safe alternative (azithromycin, doxycycline, ceftriaxone).\n"
        "4. Communicate clearly to the prescriber.\n\n"
        "Actions: get_patient_data, get_lab_results, check_formulary, "
        "check_antibiogram, reject_prescription, modify_prescription, send_message."
    ),
    prescriptions=[_rx("RX-001","P001","DR-SMITH","amoxicillin",500.0,"PO","q8h",7,"community-acquired pneumonia")],
    task_context=dict(allergies=["penicillin"],
                      correct_drugs=["azithromycin","doxycycline","levofloxacin","ceftriaxone"],
                      unsafe_drugs=["amoxicillin","penicillin","ampicillin"],
                      max_steps=20),
    grader=_grade_task1,
)

# ── TASK 2 — MEDIUM ──────────────────────────────────────────────────────────

def _grade_task2(decisions_made, messages_sent):
    if not decisions_made: return 0.0
    last = decisions_made[-1]
    atype = last.get("action_type","")
    drug  = last.get("drug","").lower()
    has_msg = len(messages_sent) > 0
    checked_lab = last.get("checked_lab", False)
    checked_abg = last.get("checked_antibiogram", False)
    CORRECT = {"nitrofurantoin","fosfomycin","trimethoprim-sulfamethoxazole"}
    FLUORO  = {"ciprofloxacin","levofloxacin"}
    if atype == "approve_prescription":      return 0.25
    if atype == "modify_prescription":
        if drug in CORRECT:
            score = 0.60
            if checked_lab: score += 0.15
            if checked_abg: score += 0.15
            if has_msg:     score += 0.10
            return round(min(1.0, score), 4)
        elif drug in FLUORO: return 0.15
    if atype == "reject_prescription":
        reason = last.get("reason","").lower()
        score = 0.50 if ("resist" in reason or "susceptib" in reason) else 0.25
        if has_msg: score += 0.15
        return round(score, 4)
    return 0.0

TASK_2 = Task(
    task_id="task_2", difficulty="medium",
    description=(
        "MEDIUM — Resistance-Guided De-escalation\n\n"
        "Patient P002 (34F) has a urinary tract infection.\n"
        "Prescriber ordered CIPROFLOXACIN 500mg PO q12h × 5 days.\n\n"
        "Hospital antibiogram: E. coli ciprofloxacin susceptibility = 62% (below 80% threshold).\n"
        "Lab culture is available. You must check resistance data before approving.\n\n"
        "Your job:\n"
        "1. Check lab results and antibiogram.\n"
        "2. If resistance confirmed, modify to appropriate narrow-spectrum agent.\n"
        "3. Prefer low-cost agent suitable for uncomplicated UTI (nitrofurantoin).\n"
        "4. Notify prescriber with rationale.\n\n"
        "Actions: get_patient_data, get_lab_results, check_formulary, "
        "check_antibiogram, approve_prescription, modify_prescription, send_message."
    ),
    prescriptions=[_rx("RX-002","P002","DR-JONES","ciprofloxacin",500.0,"PO","q12h",5,"uncomplicated urinary tract infection")],
    task_context=dict(allergies=[],
                      correct_drugs=["nitrofurantoin","fosfomycin","trimethoprim-sulfamethoxazole"],
                      unsafe_drugs=["ciprofloxacin"],
                      max_steps=20),
    grader=_grade_task2,
)

# ── TASK 3 — HARD ────────────────────────────────────────────────────────────

def _grade_task3(decisions_made, messages_sent):
    if not decisions_made: return 0.0
    score = 0.0
    has_msg     = len(messages_sent) > 0
    has_followup = any(d.get("action_type") == "follow_up" for d in decisions_made)
    consulted   = any(d.get("action_type") in ("get_patient_data","get_lab_results") for d in decisions_made)
    vanco = next((d for d in decisions_made if "vancomycin" in d.get("drug","").lower()), None)
    mero  = next((d for d in decisions_made if "meropenem"  in d.get("drug","").lower()), None)
    # Allergy violations
    for d in decisions_made:
        drug = d.get("drug","").lower()
        if any(a in drug for a in ["sulfonamide","fluoroquinolon","ciprofloxacin","levofloxacin"]):
            score -= 0.50
    # Vancomycin renal adjustment
    if vanco:
        atype = vanco.get("action_type","")
        freq  = vanco.get("frequency","").lower()
        dose  = vanco.get("dose_mg", 1000.0)
        adjusted = "q24h" in freq or "q48h" in freq or dose <= 750.0
        if atype == "modify_prescription" and adjusted:
            score += 0.30
        elif atype == "approve_prescription" and not adjusted:
            score -= 0.25
    # Meropenem (restricted — should be approved with justification)
    if mero:
        atype = mero.get("action_type","")
        if atype in ("approve_prescription","modify_prescription"):
            score += 0.25
        elif atype == "reject_prescription":
            score -= 0.20
    # Info gathering
    if consulted:  score += 0.20
    if has_msg:    score += 0.15
    if has_followup: score += 0.10
    return round(max(0.0, min(1.0, score)), 4)

TASK_3 = Task(
    task_id="task_3", difficulty="hard",
    description=(
        "HARD — ICU Multi-Drug Stewardship\n\n"
        "Patient P003 (55M) is in the ICU with confirmed MRSA bacteremia.\n"
        "Known allergies: SULFONAMIDES, FLUOROQUINOLONES.\n"
        "Severe renal impairment: eGFR = 22 mL/min. Immunosuppressed (transplant).\n\n"
        "Two active prescriptions await review:\n"
        "  (A) VANCOMYCIN 1000mg IV q12h × 14 days — MRSA coverage\n"
        "  (B) MEROPENEM 1g IV q8h × 7 days — gram-negative coverage (RESTRICTED)\n\n"
        "Your job:\n"
        "1. Review EHR + labs before any decision.\n"
        "2. Adjust vancomycin dose/frequency for severe renal impairment.\n"
        "3. Evaluate meropenem restriction waiver — justify or reject.\n"
        "4. Document clinical rationale for each decision.\n"
        "5. Communicate to ICU team.\n"
        "6. Schedule follow-up review.\n\n"
        "Actions: get_patient_data, get_lab_results, check_formulary, check_antibiogram, "
        "approve_prescription, modify_prescription, reject_prescription, send_message, follow_up."
    ),
    prescriptions=[
        _rx("RX-003A","P003","DR-ICU","vancomycin",1000.0,"IV","q12h",14,"MRSA bacteremia"),
        _rx("RX-003B","P003","DR-ICU","meropenem", 1000.0,"IV","q8h", 7, "gram-negative coverage — restricted"),
    ],
    task_context=dict(allergies=["sulfonamides","fluoroquinolones"],
                      correct_drugs=["vancomycin","daptomycin","meropenem","piperacillin-tazobactam"],
                      unsafe_drugs=["ciprofloxacin","levofloxacin","trimethoprim-sulfamethoxazole"],
                      max_steps=20),
    grader=_grade_task3,
)

# ── Registry ─────────────────────────────────────────────────────────────────

ALL_TASKS: Dict[str, Task] = {
    "task_1": TASK_1,
    "task_2": TASK_2,
    "task_3": TASK_3,
}

def get_task(task_id: str) -> Optional[Task]:
    return ALL_TASKS.get(task_id)

def list_tasks() -> List[str]:
    return list(ALL_TASKS.keys())
