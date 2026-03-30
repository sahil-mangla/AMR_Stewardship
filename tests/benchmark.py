"""
benchmark.py — Layer 3 E2E Benchmarking.
Compares Random vs. Rule-based (Stewardship) agents and verifies grader consistency.
"""

import sys, os, random
import numpy as np
from typing import Dict, List

# Ensure env is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.asp_env import ASPEnv
from env.models import Action, ActionType

class RandomAgent:
    def act(self, obs) -> Action:
        atype = random.choice([ActionType.APPROVE_PRESCRIPTION, ActionType.MODIFY_PRESCRIPTION, 
                               ActionType.SEND_MESSAGE, ActionType.NOOP])
        return Action(atype, parameters={"drug": "amoxicillin", "dose_mg": 500.0, "message": "hello"})

class RuleBasedAgent:
    """Follows basic stewardship: Get data -> Get lab -> Narrow therapy -> Send message."""
    def act(self, obs) -> Action:
        if not obs.patient_data:
            return Action(ActionType.GET_PATIENT_DATA)
        if hasattr(obs, 'lab_results') and not obs.lab_results:
            return Action(ActionType.GET_LAB_RESULTS)
        
        # If we have lab results and a pending prescription, narrow it
        if obs.lab_results and obs.pending_prescriptions[0].status == "pending":
            sensitive = obs.lab_results[0].sensitive_to
            if sensitive:
                target = sensitive[0]
            elif obs.lab_results[0].intermediate_to:
                target = obs.lab_results[0].intermediate_to[0]
            else:
                target = "meropenem"
                
            from env.generator import DRUG_PROPERTIES
            dproc = DRUG_PROPERTIES.get(target, {"base_dose": 1000.0, "freq": "q12h"})
            
            return Action(ActionType.MODIFY_PRESCRIPTION, 
                          parameters={"drug": target, "dose_mg": dproc.get("base_dose", 1000.0), 
                                      "frequency": dproc.get("freq", "q12h")})
        
        # If we modified, send a message
        if obs.decisions_made and not obs.messages_sent:
            return Action(ActionType.SEND_MESSAGE, 
                          parameters={"message": "Narrowing therapy per culture results.", "recipient_id": "DR-SMITH"})
        
        return Action(ActionType.NOOP)

def run_benchmark(agent_class, num_episodes=50):
    env = ASPEnv(task_id="dynamic")
    rewards = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent_class().act(obs)
            obs, r, done, trunc, info = env.step(action)
            ep_reward += r
        rewards.append(ep_reward)
    
    return np.mean(rewards), np.std(rewards)

def test_grader_consistency():
    """Verify that Safety vs Cost graders are consistent on stewardship paths."""
    from env.reward import SAFETY_GRADER, COST_GRADER
    env = ASPEnv(task_id="dynamic")
    obs, info = env.reset(seed=42)
    
    # Simulate a correct de-escalation action
    action = Action(ActionType.MODIFY_PRESCRIPTION, parameters={"drug": "amoxicillin", "dose_mg": 500.0})
    
    safety_score = SAFETY_GRADER.evaluate(obs, action, info["task_context"])
    cost_score = COST_GRADER.evaluate(obs, action, info["task_context"])
    
    print(f"\nGrader Consistency Check:")
    print(f"  Safety Score: {safety_score}")
    print(f"  Cost Score: {cost_score}")
    
    # In a correct path, safety should be >= 0 and cost should be >= 0 (unless drug is expensive)
    return safety_score >= 0 and cost_score >= 0

if __name__ == "__main__":
    print("Running Benchmarks...")
    
    random_mean, random_std = run_benchmark(RandomAgent, 20)
    print(f"Random Agent: Mean Reward = {random_mean:.4f} (std={random_std:.4f})")
    
    rule_mean, rule_std = run_benchmark(RuleBasedAgent, 20)
    print(f"Rule-Based Agent: Mean Reward = {rule_mean:.4f} (std={rule_std:.4f})")
    
    consistency = test_grader_consistency()
    print(f"Grader Consistency: {'PASS' if consistency else 'FAIL'}")
    
    assert rule_mean > random_mean, "Baseline Failure: Rule-based agent should outperform random agent."
