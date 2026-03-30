"""
inference.py — Baseline inference script for the Hospital ASP Coordinator Environment.

Competition requirements satisfied:
  ✓ Uses OpenAI API client for all LLM calls
  ✓ Reads credentials from environment variables: API_BASE_URL, HF_TOKEN, MODEL_NAME
  ✓ Runs all 3 tasks and produces reproducible baseline scores
  ✓ Named 'inference.py' in the root directory
  ✓ Completes in < 20 minutes on 2vCPU / 8GB RAM

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export HF_TOKEN="hf_..."
    export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    python inference.py

Output:
    Baseline scores for task_1, task_2, task_3 printed to stdout.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — all from environment variables (competition requirement)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY:      str = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
MODEL_NAME:   str = os.getenv("MODEL_NAME", "gpt-4o")

ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS   = 15
TEMPERATURE = 0.1
MAX_TOKENS  = 400

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert hospital pharmacist working in an Antimicrobial Stewardship Program (ASP).
Your job is to review antibiotic prescription requests and make safe, evidence-based decisions.

At each step you must respond with EXACTLY ONE JSON action. No prose, no markdown. Just JSON.

Available action_types:
  - "list_pending"           : see all pending prescriptions
  - "get_patient_data"       : {"patient_id": "P001"}
  - "get_lab_results"        : {"patient_id": "P001"}
  - "check_formulary"        : {"drug": "vancomycin"}
  - "check_antibiogram"      : {"organism": "E. coli"}
  - "approve_prescription"   : {"prescription_id": "RX-001"}
  - "modify_prescription"    : {"prescription_id": "RX-001", "drug": "azithromycin", "dose_mg": 500, "frequency": "q24h", "duration_days": 5}
  - "reject_prescription"    : {"prescription_id": "RX-001", "reason": "penicillin allergy"}
  - "send_message"           : {"recipient_id": "DR-SMITH", "message": "...", "prescription_id": "RX-001"}
  - "follow_up"              : {"patient_id": "P001"}
  - "noop"                   : {}

Good clinical workflow:
1. List pending prescriptions
2. Get patient data (check allergies, renal function)
3. Get lab results (organism, sensitivities)
4. Check antibiogram (hospital resistance rates)
5. Check formulary (cost, availability, alternatives)
6. Make your decision (approve/modify/reject)
7. Send a clear message to the prescriber
8. Follow up on complex cases

CRITICAL SAFETY RULES:
- NEVER prescribe a drug that matches a patient allergy
- ALWAYS adjust vancomycin/aminoglycoside doses for renal impairment (eGFR < 30)
- PREFER narrow-spectrum agents when lab sensitivities confirm coverage
- RESTRICTED drugs (vancomycin, meropenem, linezolid) require documented justification

Respond with ONLY this JSON structure:
{"action_type": "...", "parameters": {...}}
""").strip()


# ---------------------------------------------------------------------------
# HTTP client for the environment
# ---------------------------------------------------------------------------

class EnvClient:
    """Simple HTTP client for the ASP environment REST API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str) -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/reset", json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, task_id: str, action_type: str, parameters: Dict) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/step",
            json={"task_id": task_id, "action_type": action_type, "parameters": parameters},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def grade(self, task_id: str) -> float:
        r = requests.post(f"{self.base_url}/grade", json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        return r.json()["score"]

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, messages: List[Dict]) -> str:
    """Call the LLM and return raw response text."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"  [LLM ERROR] {exc}")
        return '{"action_type": "noop", "parameters": {}}'


def parse_action(text: str) -> Tuple[str, Dict]:
    """Extract action_type and parameters from LLM response."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        return data.get("action_type", "noop"), data.get("parameters", {})
    except json.JSONDecodeError:
        pass

    # Try to find JSON object inside text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return data.get("action_type", "noop"), data.get("parameters", {})
        except json.JSONDecodeError:
            pass

    return "noop", {}


def obs_to_text(obs_dict: Dict) -> str:
    """Summarise the observation as a text block for the LLM."""
    lines = [
        f"Step: {obs_dict.get('step', '?')} / {obs_dict.get('max_steps', '?')}",
        f"Task: {obs_dict.get('task_id', '?')}",
        f"Task description:\n{obs_dict.get('task_description', '')}",
        "",
        "--- PENDING PRESCRIPTIONS ---",
    ]
    for rx in obs_dict.get("pending_prescriptions", []):
        lines.append(
            f"  [{rx['id']}] {rx['drug']} {rx['dose_mg']}mg {rx['route']} "
            f"{rx['frequency']} × {rx['duration_days']}d | "
            f"Status: {rx['status']} | Indication: {rx['indication']}"
        )

    if obs_dict.get("patient_data"):
        pd = obs_dict["patient_data"]
        lines += [
            "",
            "--- PATIENT DATA ---",
            f"  Age: {pd['age']}  Weight: {pd['weight_kg']}kg",
            f"  eGFR: {pd['renal_function_egfr']}  Hepatic: {pd['hepatic_function']}",
            f"  Allergies: {pd['allergies']}",
            f"  Diagnosis: {pd['diagnosis']}  Severity: {pd['severity']}  ICU: {pd['icu_admitted']}",
            f"  Current meds: {pd['current_medications']}",
        ]

    if obs_dict.get("lab_results"):
        lines.append("\n--- LAB RESULTS ---")
        for lr in obs_dict["lab_results"]:
            lines += [
                f"  Organism: {lr['organism']} ({lr['specimen_type']})",
                f"  Sensitive: {lr['sensitive_to']}",
                f"  Resistant: {lr['resistant_to']}",
            ]

    if obs_dict.get("formulary_info"):
        fi = obs_dict["formulary_info"]
        lines += [
            "",
            "--- FORMULARY ---",
            f"  {fi['drug']}: available={fi['available']} cost=${fi['cost_per_day_usd']}/day "
            f"restricted={fi['restricted']} alternatives={fi['alternatives']}",
        ]

    if obs_dict.get("antibiogram_data"):
        lines.append("\n--- ANTIBIOGRAM ---")
        for ae in obs_dict["antibiogram_data"]:
            lines.append(
                f"  {ae['organism']} / {ae['drug']}: {ae['susceptibility_pct']}% susceptible"
            )

    if obs_dict.get("last_action_result"):
        lines += [
            "",
            f"Last action result: {obs_dict['last_action_result']}",
            f"Error: {obs_dict.get('last_action_error', False)}",
        ]

    lines.append(f"\nCumulative reward so far: {obs_dict.get('episode_reward_so_far', 0.0):.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    llm_client: OpenAI,
    env_client: EnvClient,
    task_id: str,
) -> float:
    """
    Run one full episode for a task.
    Returns the grader score (0.0–1.0).
    """
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'='*60}")

    # Reset environment
    reset_resp = env_client.reset(task_id)
    obs = reset_resp["observation"]

    conversation: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    done = False
    step = 0

    while not done and step < MAX_STEPS:
        step += 1
        obs_text = obs_to_text(obs)
        user_msg = {"role": "user", "content": obs_text}
        conversation.append(user_msg)

        # Trim conversation to last 6 turns to keep context manageable
        context = [conversation[0]] + conversation[-6:]

        raw = call_llm(llm_client, context)
        print(f"  Step {step} | LLM raw: {raw[:120]}...")

        action_type, parameters = parse_action(raw)
        print(f"  Step {step} | Action: {action_type} | Params: {parameters}")

        conversation.append({"role": "assistant", "content": raw})

        # Send action to environment
        try:
            step_resp = env_client.step(task_id, action_type, parameters)
        except Exception as exc:
            print(f"  [ENV ERROR] {exc} — falling back to noop")
            step_resp = env_client.step(task_id, "noop", {})

        obs  = step_resp["observation"]
        done = step_resp["done"]
        print(f"  Step {step} | Reward: {step_resp['reward']:+.4f} | Done: {done}")
        print(f"  Result: {obs.get('last_action_result', '')}")

        if obs.get("last_action_error"):
            print(f"  [ACTION ERROR]")

    # Get grader score
    score = env_client.grade(task_id)
    print(f"\n  GRADER SCORE for {task_id}: {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Main — run all 3 tasks and report baseline scores
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable.")
        sys.exit(1)

    print(f"Model:       {MODEL_NAME}")
    print(f"API base:    {API_BASE_URL}")
    print(f"Environment: {ENV_BASE_URL}")

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = EnvClient(base_url=ENV_BASE_URL)

    # Wait for environment to be ready
    print("\nWaiting for environment server...")
    for attempt in range(12):
        if env_client.health():
            print("Environment server is healthy.")
            break
        print(f"  Attempt {attempt + 1}/12 — retrying in 5s...")
        time.sleep(5)
    else:
        print("ERROR: Environment server did not become healthy.")
        sys.exit(1)

    # Run all tasks
    scores: Dict[str, float] = {}
    for task_id in ["task_1", "task_2", "task_3"]:
        try:
            score = run_episode(llm_client, env_client, task_id)
            scores[task_id] = score
        except Exception as exc:
            print(f"ERROR running {task_id}: {exc}")
            scores[task_id] = 0.0
        # Brief pause between episodes to avoid rate limits
        time.sleep(2)

    # Summary
    print("\n" + "="*60)
    print("  BASELINE SCORES SUMMARY")
    print("="*60)
    for task_id, score in scores.items():
        task = {"task_1": "easy", "task_2": "medium", "task_3": "hard"}[task_id]
        print(f"  {task_id} ({task:6s}): {score:.4f}")
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  Average         : {avg:.4f}")
    print("="*60)

    # Write scores to file for reproducibility
    with open("baseline_scores.json", "w") as f:
        json.dump({"scores": scores, "average": avg, "model": MODEL_NAME}, f, indent=2)
    print("\nBaseline scores written to baseline_scores.json")


if __name__ == "__main__":
    main()
