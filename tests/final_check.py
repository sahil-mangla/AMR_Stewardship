"""
final_check.py — Manual environment verification script.
Tests the environment API and graders without requiring an LLM.
"""

import requests
import time

BASE_URL = "http://localhost:7860"

def test_task_1_grading():
    print("\n--- Testing Task 1 (Allergy) ---")
    # 1. Reset
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": "task_1"})
    obs = r.json()["observation"]
    print(f"Initial Obs: {obs['task_description']}")
    
    # 2. Get data
    requests.post(f"{BASE_URL}/step", json={"task_id": "task_1", "action_type": "get_patient_data", "parameters": {}})
    
    # 3. Correct Action: Reject due to allergy
    r = requests.post(f"{BASE_URL}/step", json={
        "task_id": "task_1", 
        "action_type": "reject_prescription", 
        "parameters": {"prescription_id": "RX-T1-001", "reason": "Severe penicillin allergy"}
    })
    print(f"Step Result: {r.json()['observation']['last_action_result']}")
    
    # 4. Grade
    r = requests.post(f"{BASE_URL}/grade", json={"task_id": "task_1"})
    score = r.json()["score"]
    print(f"Grader Score: {score}")
    assert score == 1.0, f"Expected 1.0, got {score}"

def test_task_2_grading():
    print("\n--- Testing Task 2 (Resistance) ---")
    # 1. Reset
    requests.post(f"{BASE_URL}/reset", json={"task_id": "task_2"})
    
    # 2. Sequence
    requests.post(f"{BASE_URL}/step", json={"task_id": "task_2", "action_type": "get_patient_data"})
    requests.post(f"{BASE_URL}/step", json={"task_id": "task_2", "action_type": "get_lab_results"})
    requests.post(f"{BASE_URL}/step", json={"task_id": "task_2", "action_type": "check_antibiogram"})
    
    # 3. Correct Action: Modify to nitrofurantoin
    requests.post(f"{BASE_URL}/step", json={
        "task_id": "task_2", 
        "action_type": "modify_prescription", 
        "parameters": {"drug": "nitrofurantoin", "dose_mg": 100}
    })
    
    # 4. Grade
    r = requests.post(f"{BASE_URL}/grade", json={"task_id": "task_2"})
    score = r.json()["score"]
    print(f"Grader Score: {score}")
    assert score == 1.0, f"Expected 1.0, got {score}"

if __name__ == "__main__":
    try:
        test_task_1_grading()
        test_task_2_grading()
        print("\n✅ All manual environment checks passed.")
    except Exception as e:
        print(f"\n❌ Check failed: {e}")
