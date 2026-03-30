"""
generator.py — Procedural Clinical Case Generator.
Generates realistic patient scenarios with deliberate clinical errors for RL agents to fix.
"""

from __future__ import annotations
import random, time
from typing import Dict, List, Optional, Any
from env.models import (
    PrescriptionRequest, PatientData, LabResult, Severity, PrescriptionStatus,
)

# ── Medical Knowledge Base ──────────────────────────────────────────────────

INDICATIONS = {
    "uncomplicated urinary tract infection": {
        "typical_organisms": ["Escherichia coli", "Klebsiella pneumoniae"],
        "severity": Severity.MILD,
        "empiric_choice": ["nitrofurantoin", "trimethoprim-sulfamethoxazole", "fosfomycin"],
        "escalation_choice": ["ciprofloxacin", "ceftriaxone"],
        "recommended_duration": 5,
    },
    "community-acquired pneumonia": {
        "typical_organisms": ["Streptococcus pneumoniae", "Haemophilus influenzae"],
        "severity": Severity.MODERATE,
        "empiric_choice": ["ceftriaxone", "azithromycin", "doxycycline"],
        "escalation_choice": ["levofloxacin", "moxifloxacin"],
        "recommended_duration": 7,
    },
    "hospital-acquired pneumonia": {
        "typical_organisms": ["Pseudomonas aeruginosa", "Staphylococcus aureus", "Klebsiella pneumoniae"],
        "severity": Severity.SEVERE,
        "empiric_choice": ["piperacillin-tazobactam", "cefepime"],
        "escalation_choice": ["meropenem", "vancomycin"],
        "recommended_duration": 10,
    },
    "skin and soft tissue infection": {
        "typical_organisms": ["Staphylococcus aureus", "Streptococcus pyogenes"],
        "severity": Severity.MILD,
        "empiric_choice": ["oxacillin", "cefazolin", "clindamycin"],
        "escalation_choice": ["vancomycin", "daptomycin"],
        "recommended_duration": 7,
    },
    "sepsis — unknown source": {
        "typical_organisms": ["Escherichia coli", "Staphylococcus aureus", "Pseudomonas aeruginosa"],
        "severity": Severity.CRITICAL,
        "empiric_choice": ["piperacillin-tazobactam", "vancomycin"],
        "escalation_choice": ["meropenem"],
        "recommended_duration": 14,
    }
}

DRUG_PROPERTIES = {
    "vancomycin": {"spectrum": "broad-gram-positive", "renal_adjust": True, "base_dose": 1500, "freq": "q12h", "mg_kg_limit": 15.0},
    "gentamicin": {"spectrum": "narrow-gram-negative", "renal_adjust": True, "base_dose": 400, "freq": "q24h", "mg_kg_limit": 5.0},
    "meropenem": {"spectrum": "ultra-broad", "renal_adjust": True, "base_dose": 1000, "freq": "q8h"},
    "piperacillin-tazobactam": {"spectrum": "broad", "renal_adjust": True, "base_dose": 4500, "freq": "q6h"},
    "ceftriaxone": {"spectrum": "narrow", "renal_adjust": False, "base_dose": 1000, "freq": "q24h"},
    "nitrofurantoin": {"spectrum": "narrow", "renal_adjust": True, "base_dose": 100, "freq": "q12h"},
    "ciprofloxacin": {"spectrum": "medium", "renal_adjust": True, "base_dose": 500, "freq": "q12h"},
    "amoxicillin": {"spectrum": "narrow", "renal_adjust": False, "base_dose": 500, "freq": "q8h"},
}

ALLERGIES_MAP = {
    "penicillin": ["penicillin", "amoxicillin", "ampicillin", "piperacillin-tazobactam"],
    "sulfonamides": ["trimethoprim-sulfamethoxazole"],
    "fluoroquinolones": ["ciprofloxacin", "levofloxacin", "moxifloxacin"],
}

CROSS_REACTIVITY_MAP = {
    "penicillin": {"cephalosporins": 0.10, "carbapenems": 0.05}
}

COVERAGE_GROUPS = {
    "anaerobic": ["metronidazole", "clindamycin", "piperacillin-tazobactam", "meropenem"],
    "mrsa": ["vancomycin", "daptomycin", "linezolid", "ceftaroline"],
    "pseudomonas": ["piperacillin-tazobactam", "cefepime", "meropenem", "ceftazidime"]
}

# ── Generator Logic ─────────────────────────────────────────────────────────

class CaseGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)
        self.patient_count = 0

    def generate_case(self, resistance_engine: Any, seed: Optional[int] = None, task_id: Optional[str] = None) -> Dict[str, Any]:
        if seed is not None:
            self.random.seed(seed)
            
        tid = task_id or str(seed)

        # ── TASK 1: Allergy (Easy) ──────────────────────────────────────────
        if "task_1" in tid or seed == 1:
             return self._generate_task_1(resistance_engine)

        # ── TASK 2: Resistance (Medium) ─────────────────────────────────────
        if "task_2" in tid or seed == 2:
             return self._generate_task_2(resistance_engine)

        # ── TASK 3: ICU (Hard) ──────────────────────────────────────────────
        if "task_3" in tid or seed == 3:
             return self._generate_task_3(resistance_engine)

        # ── Default: Dynamic / Random ───────────────────────────────────────
        self.patient_count += 1
        pid = f"DP-{self.random.randint(1000, 9999):04d}-{self.patient_count:02d}"
        
        # 1. Select Indication & Organism
        indication_name = self.random.choice(list(INDICATIONS.keys()))
        ind_data = INDICATIONS[indication_name]
        organism = self.random.choice(ind_data["typical_organisms"])
        severity = ind_data["severity"]

        # 2. Patient Data
        age = self.random.randint(18, 90)
        weight = self.random.uniform(50.0, 120.0)
        # Randomly lower renal function for elderly patients
        egfr = self.random.uniform(60, 120) if age < 65 else self.random.uniform(15, 80)
        renal_status = "impaired" if egfr < 60 else "normal"
        
        # Random allergies
        allergies = []
        if self.random.random() < 0.2:
            allergies.append(self.random.choice(list(ALLERGIES_MAP.keys())))

        patient = PatientData(
            patient_id=pid, age=age, weight_kg=round(weight, 1),
            renal_function_egfr=round(egfr, 1), hepatic_function="normal",
            allergies=allergies, current_medications=[],
            diagnosis=indication_name, severity=severity,
            icu_admitted=(severity in [Severity.SEVERE, Severity.CRITICAL])
        )

        # 3. Lab Results
        # Get susceptibility from the local resistance engine
        antibiogram = resistance_engine.get_antibiogram(organism)
        sensitive_to = [a["drug"] for a in antibiogram if a["susceptibility_pct"] >= 80.0]
        resistant_to = [a["drug"] for a in antibiogram if a["susceptibility_pct"] < 60.0]
        intermediate_to = [a["drug"] for a in antibiogram if 60.0 <= a["susceptibility_pct"] < 80.0]

        lab_result = LabResult(
            patient_id=pid, organism=organism, specimen_type="blood/urine/sputum",
            sensitive_to=sensitive_to, resistant_to=resistant_to,
            intermediate_to=intermediate_to, reported_at=time.time()
        )

        # 4. Initial Prescription (The Problem)
        # We deliberately pick a drug that is either:
        # - An allergen for this patient (10%)
        # - A cross-reactivity risk (10%)
        # - Resistant or sub-optimal in the lab result (15%)
        # - Wrong dose for renal function/weight (15%)
        # - Redundant coverage (10%)
        # - Or actually correct (40%)
        
        error_type = self.random.choices(
            ["allergy", "cross-reactivity", "resistance", "renal", "weight", "redundant", "correct"],
            weights=[0.10, 0.10, 0.15, 0.10, 0.10, 0.05, 0.40]
        )[0]

        drug = self.random.choice(ind_data["empiric_choice"] + ind_data.get("escalation_choice", []))
        dose = 1000.0
        freq = "q12h"
        duration = ind_data["recommended_duration"]

        if error_type == "allergy" and allergies:
            drug = self.random.choice(ALLERGIES_MAP[allergies[0]])
        elif error_type == "cross-reactivity" and "penicillin" in allergies:
            drug = "cefepime" # Cephalosporin risk
        elif error_type == "resistance" and resistant_to:
            drug = self.random.choice(resistant_to)
        elif error_type == "renal" and egfr < 30:
            drug = "vancomycin"
            freq = "q12h" 
        elif error_type == "weight":
            # Giving too much vancomycin for a light patient
            drug = "vancomycin"
            dose = 2000.0 # Standard dose might be too high for low weight
        elif error_type == "redundant":
            # Sepsis case with double anaerobic
            indication_name = "sepsis — unknown source"
            drug = "piperacillin-tazobactam"
            # In the environment, maybe another drug will be active or we flag this drug
        
        initial_rx = PrescriptionRequest(
            id=f"RX-{self.random.randint(1000, 9999)}",
            patient_id=pid, prescriber_id="DR-NPC",
            drug=drug, dose_mg=dose, route="IV",
            frequency=freq, duration_days=duration + (self.random.randint(2, 5) if self.random.random() < 0.2 else 0),
            indication=indication_name, status=PrescriptionStatus.PENDING,
            created_at=time.time()
        )

        # 5. Task Metadata (For Reward Engine)
        task_context = {
            "correct_drugs": sensitive_to,
            "unsafe_drugs": resistant_to + [d for a in allergies for d in ALLERGIES_MAP[a]],
            "renal_dose_needed": (egfr < 30),
            "weight_adjust_needed": drug.lower() in ["vancomycin", "gentamicin"],
            "cross_reactivity_risk": True if (error_type == "cross-reactivity") else False,
            "indication": indication_name,
            "recommended_duration": ind_data["recommended_duration"],
            "organism": organism,
            "error_type": error_type,
            "max_steps": 20
        }

    def _generate_task_1(self, resistance_engine: Any) -> Dict[str, Any]:
        # Patient P-001 (68F) with penicillin allergy prescribed amoxicillin
        pid = "P-001"
        patient = PatientData(
            patient_id=pid, age=68, weight_kg=62.0, renal_function_egfr=75.0,
            hepatic_function="normal", allergies=["penicillin"],
            current_medications=["amlodipine"], diagnosis="community-acquired pneumonia",
            severity=Severity.MODERATE, icu_admitted=False
        )
        lab = LabResult(
            patient_id=pid, organism="Streptococcus pneumoniae", specimen_type="sputum",
            sensitive_to=["ceftriaxone", "azithromycin", "levofloxacin"],
            resistant_to=["penicillin"], reported_at=time.time()
        )
        rx = PrescriptionRequest(
            id="RX-T1-001", patient_id=pid, prescriber_id="DR-INTERNAL-MED",
            drug="amoxicillin", dose_mg=500.0, route="PO", frequency="q8h",
            duration_days=7, indication="community-acquired pneumonia",
            status=PrescriptionStatus.PENDING, created_at=time.time()
        )
        context = {
            "correct_drugs": ["ceftriaxone", "azithromycin", "levofloxacin"],
            "unsafe_drugs": ["penicillin", "amoxicillin", "ampicillin"],
            "error_type": "allergy", "max_steps": 20
        }
        return {"patient": patient, "lab": lab, "prescription": rx, "context": context}

    def _generate_task_2(self, resistance_engine: Any) -> Dict[str, Any]:
        # P-002, 34F, UTI, prescribed Cipro (resistant)
        pid = "P-002"
        patient = PatientData(
            patient_id=pid, age=34, weight_kg=58.0, renal_function_egfr=110.0,
            hepatic_function="normal", allergies=[],
            current_medications=[], diagnosis="uncomplicated urinary tract infection",
            severity=Severity.MILD, icu_admitted=False
        )
        lab = LabResult(
            patient_id=pid, organism="Escherichia coli", specimen_type="urine",
            sensitive_to=["nitrofurantoin", "fosfomycin"],
            resistant_to=["ciprofloxacin", "trimethoprim-sulfamethoxazole"],
            reported_at=time.time()
        )
        rx = PrescriptionRequest(
            id="RX-T2-001", patient_id=pid, prescriber_id="DR-URGENT-CARE",
            drug="ciprofloxacin", dose_mg=500.0, route="PO", frequency="q12h",
            duration_days=5, indication="uti",
            status=PrescriptionStatus.PENDING, created_at=time.time()
        )
        context = {
            "correct_drugs": ["nitrofurantoin", "fosfomycin"],
            "unsafe_drugs": ["ciprofloxacin", "tmpsmx"],
            "error_type": "resistance", "max_steps": 20
        }
        return {"patient": patient, "lab": lab, "prescription": rx, "context": context}

    def _generate_task_3(self, resistance_engine: Any) -> Dict[str, Any]:
        # P-003, ICU, MRSA, eGFR=22
        pid = "P-003"
        patient = PatientData(
            patient_id=pid, age=55, weight_kg=85.0, renal_function_egfr=22.0,
            hepatic_function="normal", allergies=["sulfonamides", "fluoroquinolones"],
            current_medications=["norepinephrine"], diagnosis="sepsis — mrsa bacteremia",
            severity=Severity.CRITICAL, icu_admitted=True
        )
        lab = LabResult(
            patient_id=pid, organism="Staphylococcus aureus (MRSA)", specimen_type="blood",
            sensitive_to=["vancomycin", "daptomycin", "linezolid"],
            resistant_to=["oxacillin", "cefazolin"], reported_at=time.time()
        )
        # Note: In our current single-prescription environment, we'll focus on the Vancomycin adjustment
        # for task 3, but the context will note the Mero restriction.
        rx = PrescriptionRequest(
            id="RX-T3-001", patient_id=pid, prescriber_id="DR-ICU",
            drug="vancomycin", dose_mg=1500.0, route="IV", frequency="q12h",
            duration_days=14, indication="mrsa sepsis",
            status=PrescriptionStatus.PENDING, created_at=time.time()
        )
        context = {
            "correct_drugs": ["vancomycin"],
            "renal_dose_needed": True,
            "weight_adjust_needed": True,
            "error_type": "renal", "max_steps": 20
        }
        return {"patient": patient, "lab": lab, "prescription": rx, "context": context}

# Remove global GEN. Each env should instantiate its own generator or use a shared seeded one.
# For simplicity, we'll instantiate it in ASPEnv.__init__.
