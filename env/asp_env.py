"""
asp_env.py — Hospital ASP Coordinator Environment (OpenEnv compliant).
Fully Gymnasium-compliant with seeding and observation space.
"""

from __future__ import annotations
import copy, sys, os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.models import (
    Action, ActionType, Observation, PrescriptionStatus, Reward,
)
from tasks.tasks import run_grader, ALL_TASKS
from env.reward import compute_reward
from env.tools import (
    check_antibiogram, check_formulary, get_lab_results,
    get_patient_data, list_pending, send_message,
)

class ASPEnv(gym.Env):
    """Hospital Antimicrobial Stewardship Program Coordinator Environment."""
    metadata = {"render_modes": ["human", "ansi"]}
    MAX_STEPS = 20

    def __init__(self, task_id: str = "dynamic"):
        super().__init__()
        self.task_id = task_id
        self._obs: Optional[Observation] = None
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._episode_log: List[Dict] = []
        self._current_context: Dict[str, Any] = {}
        self._npc = None
        
        from env.generator import CaseGenerator
        self._generator = CaseGenerator()
        from env.resistance import ResistanceEngine
        self._resistance_engine = ResistanceEngine()

        # ── Gymnasium Spaces ────────────────────────────────────────────────
        # Placeholder space for OpenEnv compliance (OpenEnv usually uses Dict obs)
        self.observation_space = spaces.Dict({
            "step": spaces.Discrete(self.MAX_STEPS + 1),
            "task_id": spaces.Text(min_length=0, max_length=100),
            "episode_reward": spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(len(ActionType)),
            "parameters": spaces.Dict({}), # Dynamic params
        })

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Observation, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Ensure our generator uses the gym np_random if needed
        # (For now, we'll just seed the global random for simplicity in this prototype)
        import random
        if seed is not None:
            random.seed(seed)
        
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._episode_log = []
        
        # 1. Generate or load task
        case = self._generator.generate_case(self._resistance_engine, seed=seed, task_id=self.task_id)
        patient = case["patient"]
        lab = case["lab"]
        prescription = case["prescription"]
        self._current_context = case["context"]
        
        from env.npc import get_npc_for_task
        self._npc = get_npc_for_task(self.task_id)
        
        # Internal state for tools
        self._current_patient = patient
        self._current_labs = [lab]
        
        prescriptions = [prescription]
        desc = f"PATIENT {patient.patient_id}: {patient.diagnosis.upper()}"
        
        self._obs = Observation(
            step=0, task_id=self.task_id,
            task_description=desc,
            pending_prescriptions=prescriptions,
            last_action_result="Episode started.",
        )
        
        info = {"task_context": self._current_context}
        return self._obs, info

    def step(self, action_input: Any) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        # Handle both Action objects and dict inputs
        if isinstance(action_input, dict):
            action = Action(
                action_type=ActionType(action_input["action_type"]),
                parameters=action_input.get("parameters", {})
            )
        else:
            action = action_input

        if self._obs is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            return copy.deepcopy(self._obs), 0.0, True, False, {}

        obs_before = copy.deepcopy(self._obs)
        self._step_count += 1

        # 1. Dispatch action
        result_msg, error_flag = self._dispatch(action)

        # 2. Update observation
        new_obs = copy.deepcopy(self._obs)
        new_obs.step = self._step_count
        new_obs.last_action = action.action_type.value if hasattr(action.action_type, 'value') else action.action_type
        new_obs.last_action_result = result_msg
        new_obs.last_action_error = error_flag

        # 3. Handle selection pressure (AMR Evolution)
        if action.action_type in [ActionType.APPROVE_PRESCRIPTION, ActionType.MODIFY_PRESCRIPTION]:
            drug = action.parameters.get("drug", "")
            is_broad = drug.lower() in ["meropenem", "vancomycin", "piperacillin-tazobactam", "cefepime"]
            self._resistance_engine.record_usage(drug, is_broad)

        # 4. Check for episode termination
        truncated = self._step_count >= self.MAX_STEPS
        all_processed = all(p.status != PrescriptionStatus.PENDING for p in new_obs.pending_prescriptions)
        
        # Episode ends when all processed or max steps reached
        self._done = all_processed or truncated

        # 5. Compute reward
        reward_obj = compute_reward(obs_before, action, new_obs, self._current_context)
        self._cumulative_reward += reward_obj.total
        new_obs.episode_reward_so_far = round(self._cumulative_reward, 4)

        self._episode_log.append({
            "step": self._step_count,
            "action": action.model_dump(),
            "reward": reward_obj.model_dump(),
        })
        self._obs = new_obs

        info = {
            "reward_breakdown": reward_obj.model_dump(),
            "cumulative_reward": self._cumulative_reward,
            "step": self._step_count,
            "task_id": self.task_id,
        }
        return self._obs, reward_obj.total, self._done, truncated, info

    def state(self) -> Dict[str, Any]:
        """Return full environment state snapshot."""
        return {
            "observation": self._obs.model_dump() if self._obs else None,
            "cumulative_reward": self._cumulative_reward,
            "step_count": self._step_count,
            "done": self._done,
            "task_id": self.task_id,
        }

    def run_grader(self) -> float:
        """Run the programmatic grader for the current episode history."""
        return run_grader(self.task_id, self._episode_log)

    def render(self):
        if self._obs:
            print(f"Step {self._step_count}: {self._obs.last_action_result}")

    # ── Action dispatch ──────────────────────────────────────────────────────

    def _dispatch(self, action: Action) -> Tuple[str, bool]:
        atype  = ActionType(action.action_type) if isinstance(action.action_type, str) else action.action_type
        params = action.parameters

        if atype == ActionType.LIST_PENDING:
            r = list_pending(self._obs.pending_prescriptions)
            return r.message, not r.success

        if atype == ActionType.GET_PATIENT_DATA:
            pid = params.get("patient_id") or (self._obs.pending_prescriptions[0].patient_id if self._obs.pending_prescriptions else "")
            r = get_patient_data(pid, self._current_patient)
            if r.success: self._obs.patient_data = r.data
            return r.message, not r.success

        if atype == ActionType.GET_LAB_RESULTS:
            pid = params.get("patient_id") or (self._obs.pending_prescriptions[0].patient_id if self._obs.pending_prescriptions else "")
            r = get_lab_results(pid, self._current_labs)
            if r.success: self._obs.lab_results = r.data
            return r.message, not r.success

        if atype == ActionType.CHECK_ANTIBIOGRAM:
            org = params.get("organism") or (self._obs.lab_results[0].organism if self._obs.lab_results else "")
            r = check_antibiogram(org, self._resistance_engine)
            if r.success: self._obs.antibiogram_data = r.data
            return r.message, not r.success
            
        if atype == ActionType.CHECK_FORMULARY:
            drug = params.get("drug") or (self._obs.pending_prescriptions[0].drug if self._obs.pending_prescriptions else "")
            r = check_formulary(drug)
            if r.success: self._obs.formulary_info = r.data
            return r.message, not r.success

        if atype in (ActionType.APPROVE_PRESCRIPTION, ActionType.MODIFY_PRESCRIPTION, ActionType.REJECT_PRESCRIPTION):
            return self._handle_decision(atype, params)

        if atype == ActionType.SEND_MESSAGE:
            rx_id = params.get("prescription_id") or (self._obs.pending_prescriptions[0].id if self._obs.pending_prescriptions else "")
            last_drug = self._obs.decisions_made[-1].get("drug", "none") if self._obs.decisions_made else ""
            last_type = self._obs.decisions_made[-1].get("action_type", "none") if self._obs.decisions_made else ""
            r = send_message(params.get("recipient_id",""), params.get("message",""), rx_id, self._npc, last_type, last_drug)
            if r.success: self._obs.messages_sent.append(str(r.data))
            return r.message, not r.success

        return "Action completed.", False

    def _handle_decision(self, atype: ActionType, params: Dict) -> Tuple[str, bool]:
        rx = self._obs.pending_prescriptions[0] # Simple hack for single patient
        
        if atype == ActionType.APPROVE_PRESCRIPTION:
            rx.status = PrescriptionStatus.APPROVED
            msg = f"Rx {rx.id} APPROVED."
        elif atype == ActionType.MODIFY_PRESCRIPTION:
            rx.drug = params.get("drug", rx.drug)
            rx.dose_mg = float(params.get("dose_mg", rx.dose_mg))
            rx.frequency = params.get("frequency", rx.frequency)
            rx.status = PrescriptionStatus.MODIFIED
            msg = f"Rx {rx.id} MODIFIED to {rx.drug}."
        elif atype == ActionType.REJECT_PRESCRIPTION:
            rx.status = PrescriptionStatus.REJECTED
            msg = f"Rx {rx.id} REJECTED."
            
        self._obs.decisions_made.append({"action_type": atype.value, "drug": rx.drug, "step": self._step_count})
        return msg, False
