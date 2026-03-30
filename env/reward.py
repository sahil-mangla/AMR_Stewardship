"""
reward.py — Advanced Multi-objective reward function for the ASP environment.
Includes deep medical rules and separate specialized graders.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from env.models import Action, ActionType, Observation, Reward

# ── Medical Knowledge ───────────────────────────────────────────────────────

_NARROW = {"penicillin", "amoxicillin", "ampicillin", "oxacillin", "nafcillin",
           "cefazolin", "ceftriaxone", "azithromycin", "doxycycline", "nitrofurantoin",
           "trimethoprim-sulfamethoxazole", "fosfomycin", "clindamycin"}

_BROAD = {"meropenem", "ertapenem", "imipenem", "piperacillin-tazobactam",
          "cefepime", "vancomycin", "linezolid", "daptomycin"}

CROSS_REACTIVITY = {
    "penicillin": {"cephalosporin": 0.10, "carbapenem": 0.05}
}

REDUNDANT_GROUPS = {
    "anaerobe": ["metronidazole", "clindamycin", "piperacillin-tazobactam", "meropenem"],
    "mrsa": ["vancomycin", "daptomycin", "linezolid"],
}

# ── Helper functions ────────────────────────────────────────────────────────

def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))

# ── Specialized Graders ─────────────────────────────────────────────────────

class SafetyGrader:
    """Focuses on patient safety: allergies, toxicity, dosing."""
    def evaluate(self, obs: Observation, action: Action, context: Dict) -> float:
        score = 0.0
        params = action.parameters
        drug = (params.get("drug", "") or "").lower()
        if not drug and obs.pending_prescriptions:
            drug = obs.pending_prescriptions[0].drug.lower()
            
        # 1. Allergy check
        unsafe = [d.lower() for d in context.get("unsafe_drugs", [])]
        if drug in unsafe:
            score -= 0.60
            
        # 2. Cross-reactivity check
        if context.get("cross_reactivity_risk") and ("cef" in drug or "penem" in drug):
            score -= 0.30
            
        # 3. Weight-based dosing (High-risk drugs)
        if context.get("weight_adjust_needed") and drug in ["vancomycin", "gentamicin"]:
            dose = float(params.get("dose_mg", 0)) or obs.pending_prescriptions[0].dose_mg
            weight = obs.patient_data.weight_kg if obs.patient_data else 70.0
            mg_kg = dose / weight
            # Vancomycin target ~15mg/kg
            if drug == "vancomycin" and (mg_kg > 20 or mg_kg < 10):
                score -= 0.25
            # Gentamicin target ~5mg/kg
            if drug == "gentamicin" and (mg_kg > 7 or mg_kg < 3):
                score -= 0.25
                
        # 4. Renal adjustment
        if context.get("renal_dose_needed"):
            freq = params.get("frequency", "").lower() or obs.pending_prescriptions[0].frequency.lower()
            if "q8h" in freq or "q12h" in freq:
                score -= 0.20
                
        return _clamp(score)

class CostGrader:
    """Focuses on formulary costs and redundant coverage."""
    def evaluate(self, obs: Observation, action: Action, context: Dict) -> float:
        score = 0.0
        params = action.parameters
        drug = (params.get("drug", "") or "").lower()
        
        # 1. Redundant coverage check
        if context.get("error_type") == "redundant":
            # If the user doesn't modify or address the redundancy
            if action.action_type == ActionType.APPROVE_PRESCRIPTION:
                score -= 0.30
        
        # 2. Formulary cost
        if obs.formulary_info:
            cost = obs.formulary_info.cost_per_day_usd
            if cost > 150: score -= 0.10
            elif cost < 20: score += 0.10
            
        return _clamp(score)

# ── Main Reward Function ────────────────────────────────────────────────────

SAFETY_GRADER = SafetyGrader()
COST_GRADER   = CostGrader()

def compute_reward(obs_before: Observation, action: Action,
                   obs_after: Observation, task_context: Dict[str, Any]) -> Reward:
    current_step = obs_before.step
    max_steps    = task_context.get("max_steps", 20)
    
    ps = rs = ce = sat = pq = pen = 0.0
    notes: List[str] = []
    
    atype = ActionType(action.action_type) if isinstance(action.action_type, str) else action.action_type
    params = action.parameters

    # ── 1. Safety & Toxicity (SafetyGrader) ──
    if atype in [ActionType.APPROVE_PRESCRIPTION, ActionType.MODIFY_PRESCRIPTION]:
        safety_signal = SAFETY_GRADER.evaluate(obs_before, action, task_context)
        ps += safety_signal
        if safety_signal < 0: notes.append(f"safety failure: {safety_signal}")
    
    # ── 2. Stewardship (Resistance impact) ──
    if atype in [ActionType.APPROVE_PRESCRIPTION, ActionType.MODIFY_PRESCRIPTION]:
        drug = (params.get("drug", "") or "").lower()
        correct = [d.lower() for d in task_context.get("correct_drugs", [])]
        
        if drug in _NARROW:
            rs += 0.20
            notes.append("rs: narrow-spectrum bonus")
        elif drug in _BROAD:
            if any(d in correct for d in _NARROW):
                rs -= 0.15
                notes.append("rs: broad spectrum when narrow sensitive")
            else:
                rs += 0.10
                notes.append("rs: broad spectrum justified")

    # ── 3. Cost & Efficiency (CostGrader) ──
    if atype in [ActionType.APPROVE_PRESCRIPTION, ActionType.MODIFY_PRESCRIPTION]:
        cost_signal = COST_GRADER.evaluate(obs_before, action, task_context)
        ce += cost_signal
        if cost_signal < 0: notes.append(f"cost/redundancy failure: {cost_signal}")

    # ── 4. Communication & NPC Satisfaction ──
    if atype == ActionType.SEND_MESSAGE:
        msg = params.get("message", "").lower()
        if "renal" in msg or "allergy" in msg or "culture" in msg:
            sat += 0.15
            notes.append("sat: clinical rationale provided")
        if len(msg) < 10:
            pen -= 0.05
            notes.append("pen: weak communication")

    # ── 5. Process Quality (Once-per-tool bonus) ──
    if atype in [ActionType.GET_PATIENT_DATA, ActionType.GET_LAB_RESULTS, ActionType.CHECK_ANTIBIOGRAM]:
        # Only bonus if not already done or if result was new info
        if not obs_before.patient_data and atype == ActionType.GET_PATIENT_DATA:
            pq += 0.05
        if not obs_before.lab_results and atype == ActionType.GET_LAB_RESULTS:
            pq += 0.05
        if not obs_before.antibiogram_data and atype == ActionType.CHECK_ANTIBIOGRAM:
            pq += 0.05
    
    # Per-step efficiency penalty
    pen -= 0.01 * current_step

    # Timeliness bonus
    if atype in [ActionType.APPROVE_PRESCRIPTION, ActionType.MODIFY_PRESCRIPTION, ActionType.REJECT_PRESCRIPTION]:
        timeliness = max(0, 1.0 - (current_step / max_steps))
        pq += 0.05 * timeliness

    # ── Total Weighing ──
    # safety (40%), stewardship (25%), cost (15%), satisfaction (10%), process (10%)
    total = _clamp(0.40*ps + 0.25*rs + 0.15*ce + 0.10*sat + 0.10*pq + pen)
    
    return Reward(
        total=round(total, 4),
        patient_safety=round(ps, 4),
        resistance_stewardship=round(rs, 4),
        cost_efficiency=round(ce, 4),
        prescriber_satisfaction=round(sat, 4),
        process_quality=round(pq, 4),
        penalty=round(pen, 4),
        explanation=" | ".join(notes) if notes else "baseline"
    )
