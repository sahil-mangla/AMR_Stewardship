import pytest
from env.reward import compute_reward
from env.models import Action, ActionType, Observation, PrescriptionRequest

def test_reward_allergy_penalty():
    """Verify that prescribing a drug the patient is allergic to hits deep penalty."""
    from env.reward import SAFETY_GRADER
    context = {"unsafe_drugs": ["penicillin"], "correct_drugs": ["ceftriaxone"]}
    obs_before = Observation(step=0, task_id="test", task_description="test")
    action = Action(ActionType.APPROVE_PRESCRIPTION, parameters={"drug": "penicillin"})
    
    score = SAFETY_GRADER.evaluate(obs_before, action, context)
    assert score <= -0.50

def test_reward_weight_based_dosing():
    """Verify weight-based dosing penalty for vancomycin."""
    from env.reward import SAFETY_GRADER
    from env.models import PatientData, Severity
    patient = PatientData(patient_id="P1", age=30, weight_kg=50.0, renal_function_egfr=90.0,
                          hepatic_function="normal", allergies=[], current_medications=[],
                          diagnosis="sepsis", severity=Severity.SEVERE, icu_admitted=True)
    obs_before = Observation(step=0, task_id="test", task_description="test", patient_data=patient)
    context = {"weight_adjust_needed": True}
    
    # 2000mg for 50kg = 40mg/kg (Way too high, target 15)
    action_high = Action(ActionType.APPROVE_PRESCRIPTION, parameters={"drug": "vancomycin", "dose_mg": 2000.0})
    score_high = SAFETY_GRADER.evaluate(obs_before, action_high, context)
    assert score_high < 0.0

def test_reward_cost_efficiency():
    """Verify that cheaper drugs get better cost scores."""
    from env.reward import COST_GRADER
    from env.models import FormularyEntry
    obs_before = Observation(step=0, task_id="test", task_description="test",
                             formulary_info=FormularyEntry(drug="amoxicillin", available=True, cost_per_day_usd=5.0, restricted=False, alternatives=[]))
    action = Action(ActionType.APPROVE_PRESCRIPTION, parameters={"drug": "amoxicillin"})
    
    score = COST_GRADER.evaluate(obs_before, action, {})
    assert score >= 0.10
