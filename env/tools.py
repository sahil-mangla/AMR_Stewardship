"""
tools.py — Stateless Tool Logic for the ASP environment.
These functions are called by the environment dispatch system.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from env.models import (
    PatientData, LabResult, AntibiogramEntry, FormularyEntry, ToolResult,
)
from env.npc import PrescriberNPC

def get_patient_data(patient_id: str, context_patient: Optional[PatientData]) -> ToolResult:
    if context_patient and context_patient.patient_id == patient_id:
        return ToolResult(success=True, data=context_patient, message="EHR lookup successful.")
    return ToolResult(success=False, data=None, message=f"Patient {patient_id} not found in this unit.")

def get_lab_results(patient_id: str, context_labs: List[LabResult]) -> ToolResult:
    results = [lab for lab in context_labs if lab.patient_id == patient_id]
    if results:
        return ToolResult(success=True, data=results, message=f"Found {len(results)} lab report(s).")
    return ToolResult(success=False, data=[], message="No laboratory results available for this patient.")

def check_antibiogram(organism: str, resistance_engine: Any) -> ToolResult:
    data = resistance_engine.get_antibiogram(organism)
    if data:
        return ToolResult(success=True, data=[AntibiogramEntry(**d) for d in data], message="Antibiogram retrieved.")
    return ToolResult(success=False, data=[], message=f"No antibiogram data for {organism}.")

def check_formulary(drug: str) -> ToolResult:
    # Static formulary for this prototype
    from env.generator import DRUG_PROPERTIES
    dproc = DRUG_PROPERTIES.get(drug.lower())
    if dproc:
        entry = FormularyEntry(
            drug=drug.lower(),
            available=True,
            cost_per_day_usd=50.0 if dproc["spectrum"] == "narrow" else 200.0,
            restricted=dproc["spectrum"] == "ultra-broad",
            alternatives=["amoxicillin"] if dproc["spectrum"] != "narrow" else []
        )
        return ToolResult(success=True, data=entry, message="Formulary info found.")
    return ToolResult(success=False, data=None, message=f"Drug {drug} not in hospital formulary.")

def list_pending(prescriptions: List[Any]) -> ToolResult:
    pending = [p for p in prescriptions if p.status == "pending"]
    return ToolResult(success=True, data=pending, message=f"Retrieved {len(pending)} pending order(s).")

def send_message(recipient_id: str, message: str, prescription_id: str, 
                 npc: Optional[PrescriberNPC], action_type: str = "none", 
                 drug: str = "none") -> ToolResult:
    if not message.strip():
        return ToolResult(success=False, data=None, message="Cannot send empty message.")
    
    if npc:
        response = npc.generate_response(message, action_type, drug)
        return ToolResult(success=True, data=response, message="Message delivered and response received.")
    
    return ToolResult(success=True, data=f"[DELIVERED to {recipient_id}]", message="Message delivered.")
