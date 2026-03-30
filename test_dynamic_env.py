"""
test_dynamic_env.py — Simple verification script for the newly implemented environment features.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from env.asp_env import ASPEnv
from env.models import Action, ActionType

def run_test_episode():
    print("\n=== STARTING TEST EPISODE ===")
    env = ASPEnv(task_id="dynamic")
    obs = env.reset()
    
    print(f"Task ID: {obs.task_id}")
    print(f"Description: {obs.task_description[:100]}...")
    initial_rx = obs.pending_prescriptions[0]
    print(f"Pending Rx: {initial_rx.drug} {initial_rx.dose_mg}mg for {initial_rx.indication}")

    # Action 1: Get patient data
    print("\nAction: Getting patient data...")
    obs, reward, done, info = env.step(Action(ActionType.GET_PATIENT_DATA))
    print(f"Result: {obs.last_action_result}")
    if obs.patient_data:
        print(f"Patient eGFR: {obs.patient_data.renal_function_egfr}, Allergies: {obs.patient_data.allergies}")

    # Action 2: Get lab results
    print("\nAction: Getting lab results...")
    obs, reward, done, info = env.step(Action(ActionType.GET_LAB_RESULTS))
    print(f"Result: {obs.last_action_result}")
    if obs.lab_results:
        print(f"Organism: {obs.lab_results[0].organism}, Sensitive to: {obs.lab_results[0].sensitive_to[:3]}...")

    # Action 3: Modify prescription (Simulate a fix)
    # We'll just pick the first sensitive drug if available, or stay with the original
    fix_drug = obs.lab_results[0].sensitive_to[0] if obs.lab_results and obs.lab_results[0].sensitive_to else initial_rx.drug
    print(f"\nAction: Modifying Rx to {fix_drug} (Renal adjustment included if needed)...")
    
    obs, reward, done, info = env.step(Action(
        ActionType.MODIFY_PRESCRIPTION, 
        parameters={
            "prescription_id": initial_rx.id,
            "drug": fix_drug,
            "dose_mg": 500.0, # Lower dose for test
            "frequency": "q24h",
            "duration_days": 7
        }
    ))
    print(f"Result: {obs.last_action_result}")
    print(f"Reward for decision: {reward}")

    # Action 4: Send message to NPC
    print("\nAction: Sending message to prescriber...")
    obs, reward, done, info = env.step(Action(
        ActionType.SEND_MESSAGE,
        parameters={
            "message": "I modified the prescription based on lab results and renal function (eGFR).",
            "recipient_id": "DR-NPC"
        }
    ))
    print(f"NPC Response: {obs.messages_sent[-1]}")
    print(f"Reward for message: {reward}")

    print("\n=== EPISODE FINISHED ===")
    print(f"Final Reward: {obs.episode_reward_so_far}")
    print(f"Done: {done}")

if __name__ == "__main__":
    for i in range(3):
        run_test_episode()
