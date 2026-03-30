"""
models.py — Typed Pydantic models for the Hospital ASP Coordinator Environment.
Fully compliant with OpenEnv specification.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    GET_PATIENT_DATA     = "get_patient_data"
    GET_LAB_RESULTS      = "get_lab_results"
    CHECK_FORMULARY      = "check_formulary"
    CHECK_ANTIBIOGRAM    = "check_antibiogram"
    LIST_PENDING         = "list_pending"
    APPROVE_PRESCRIPTION = "approve_prescription"
    MODIFY_PRESCRIPTION  = "modify_prescription"
    REJECT_PRESCRIPTION  = "reject_prescription"
    SEND_MESSAGE         = "send_message"
    FOLLOW_UP            = "follow_up"
    NOOP                 = "noop"

class Severity(str, Enum):
    MILD     = "mild"
    MODERATE = "moderate"
    SEVERE   = "severe"
    CRITICAL = "critical"

class PrescriptionStatus(str, Enum):
    PENDING   = "pending"
    APPROVED  = "approved"
    MODIFIED  = "modified"
    REJECTED  = "rejected"
    ESCALATED = "escalated"

class PrescriptionRequest(BaseModel):
    id: str
    patient_id: str
    prescriber_id: str
    drug: str
    dose_mg: float
    route: str
    frequency: str
    duration_days: int
    indication: str
    status: PrescriptionStatus = PrescriptionStatus.PENDING
    created_at: float = 0.0

class PatientData(BaseModel):
    patient_id: str
    age: int
    weight_kg: float
    renal_function_egfr: float
    hepatic_function: str
    allergies: List[str]
    current_medications: List[str]
    diagnosis: str
    severity: Severity
    icu_admitted: bool

class LabResult(BaseModel):
    patient_id: str
    organism: str
    specimen_type: str
    sensitive_to: List[str]
    resistant_to: List[str]
    reported_at: float

class FormularyEntry(BaseModel):
    drug: str
    available: bool
    cost_per_day_usd: float
    restricted: bool
    alternatives: List[str]

class AntibiogramEntry(BaseModel):
    organism: str
    drug: str
    susceptibility_pct: float

class Observation(BaseModel):
    step: int
    task_id: str
    task_description: str
    pending_prescriptions: List[PrescriptionRequest] = Field(default_factory=list)
    patient_data: Optional[PatientData] = None
    lab_results: Optional[List[LabResult]] = None
    formulary_info: Optional[FormularyEntry] = None
    antibiogram_data: Optional[List[AntibiogramEntry]] = None
    last_action: Optional[str] = None
    last_action_result: Optional[str] = None
    last_action_error: bool = False
    messages_sent: List[str] = Field(default_factory=list)
    decisions_made: List[Dict[str, Any]] = Field(default_factory=list)
    episode_reward_so_far: float = 0.0
    max_steps: int = 20
    done: bool = False

class Action(BaseModel):
    action_type: ActionType
    parameters: Dict[str, Any] = Field(default_factory=dict)

class Reward(BaseModel):
    total: float
    patient_safety: float = 0.0
    resistance_stewardship: float = 0.0
    cost_efficiency: float = 0.0
    prescriber_satisfaction: float = 0.0
    process_quality: float = 0.0
    penalty: float = 0.0
    explanation: str = ""

class ToolResult(BaseModel):
    success: bool
    data: Any
    message: str
