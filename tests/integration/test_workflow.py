import pytest
from env.asp_env import ASPEnv
from env.models import Action, ActionType

def test_full_de_escalation_workflow():
    """Verify a complete workflow from admission to de-escalation."""
    env = ASPEnv(task_id="dynamic")
    obs, info = env.reset(seed=42)
    
    # 1. Gather data
    obs, r1, done, trunc, info = env.step(Action(ActionType.GET_PATIENT_DATA))
    obs, r2, done, trunc, info = env.step(Action(ActionType.GET_LAB_RESULTS))
    
    # 2. Make stewardship decision
    if not obs.lab_results or not obs.lab_results[0].sensitive_to:
        pytest.skip(f"Seed {42} generated a case without explicit sensitive drugs: {obs.lab_results}")
        
    target_drug = obs.lab_results[0].sensitive_to[0]
    obs, r3, done, trunc, info = env.step(Action(
        ActionType.MODIFY_PRESCRIPTION, 
        parameters={"drug": target_drug, "dose_mg": 500.0, "frequency": "q12h"}
    ))
    
    # 3. Communicate with NPC
    obs, r4, done, trunc, info = env.step(Action(
        ActionType.SEND_MESSAGE,
        parameters={"message": f"Narrowing therapy to {target_drug} per cultures.", "recipient_id": "DR-NPC"}
    ))
    
    # Verify the results
    assert len(obs.decisions_made) == 1
    assert len(obs.messages_sent) == 1
    assert obs.pending_prescriptions[0].status == "modified"

def test_reproducibility():
    """Verify that the same seed produces the same patient case."""
    env1 = ASPEnv(task_id="dynamic")
    obs1, _ = env1.reset(seed=100)
    
    env2 = ASPEnv(task_id="dynamic")
    obs2, _ = env2.reset(seed=100)
    
    # 1. Gather data to populate patient_data
    obs1, _, _, _, _ = env1.step(Action(ActionType.GET_PATIENT_DATA))
    obs2, _, _, _, _ = env2.step(Action(ActionType.GET_PATIENT_DATA))
    
    assert obs1.patient_data.patient_id == obs2.patient_data.patient_id
    assert obs1.pending_prescriptions[0].drug == obs2.pending_prescriptions[0].drug
