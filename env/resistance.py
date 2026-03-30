"""
resistance.py — Global Antimicrobial Resistance (AMR) Evolution Engine.
Simulates hospital-wide susceptibility trends that change based on usage.
"""

from __future__ import annotations
from typing import Dict, List

# Baseline susceptibilities aligned with WHO GLASS data (approximate)
DEFAULT_SUSCEPTIBILITY = {
    "Escherichia coli": {
        "ciprofloxacin": 0.58,  # High fluoroquinolone resistance
        "nitrofurantoin": 0.94,
        "ceftriaxone": 0.72,  # ESBL prevalence
        "trimethoprim-sulfamethoxazole": 0.65,
        "fosfomycin": 0.96,
        "meropenem": 0.99,
        "piperacillin-tazobactam": 0.88,
    },
    "Staphylococcus aureus": {
        "oxacillin": 0.60,  # MRSA rate ~40%
        "vancomycin": 1.00,
        "clindamycin": 0.72,
        "daptomycin": 0.99,
        "linezolid": 0.99,
        "trimethoprim-sulfamethoxazole": 0.85,
    },
    "Streptococcus pneumoniae": {
        "penicillin": 0.65,
        "ceftriaxone": 0.90,
        "levofloxacin": 0.98,
        "azithromycin": 0.65,
        "doxycycline": 0.72,
    },
    "Pseudomonas aeruginosa": {
        "piperacillin-tazobactam": 0.75,
        "cefepime": 0.78,
        "meropenem": 0.72,
        "ciprofloxacin": 0.62,
        "tobramycin": 0.88,
        "aztreonam": 0.68,
    },
    "Klebsiella pneumoniae": {
        "ceftriaxone": 0.65,
        "meropenem": 0.88,
        "piperacillin-tazobactam": 0.75,
        "ciprofloxacin": 0.68,
    }
}

BIOLOGICAL_MINIMUM = 0.05
BIOLOGICAL_MAXIMUM = 1.00

class ResistanceEngine:
    def __init__(self, susceptibility: Dict[str, Dict[str, float]] = None):
        self.susceptibility = susceptibility or DEFAULT_SUSCEPTIBILITY
        self.evolution_rate = 0.005  # Decay rate per broad-spectrum use

    def get_susceptibility(self, organism: str, drug: str) -> float:
        org_data = self.susceptibility.get(organism, {})
        return org_data.get(drug, 0.5)  # Default to 50% if unknown

    def record_usage(self, drug: str, is_broad_spectrum: bool):
        """
        Simulate selection pressure.
        If a broad-spectrum drug is used, decrease susceptibility across all organisms 
        that usually react to it.
        """
        if not is_broad_spectrum:
            return

        for organism in self.susceptibility:
            if drug in self.susceptibility[organism]:
                # Decrease susceptibility (evolution towards resistance)
                current = self.susceptibility[organism][drug]
                new_val = max(BIOLOGICAL_MINIMUM, current - self.evolution_rate)
                self.susceptibility[organism][drug] = round(new_val, 4)

    def trigger_outbreak(self, organism: str, drug: str):
        """Simulate a sudden hospital outbreak of a highly resistant strain."""
        if organism in self.susceptibility and drug in self.susceptibility[organism]:
            self.susceptibility[organism][drug] = BIOLOGICAL_MINIMUM
            return True
        return False

    def validate_stability(self) -> Dict[str, Any]:
        """Check for biological drift anomalies (e.g. susceptibilities > 1.0)."""
        anomalies = []
        for org, drugs in self.susceptibility.items():
            for drug, val in drugs.items():
                if val < BIOLOGICAL_MINIMUM or val > BIOLOGICAL_MAXIMUM:
                    anomalies.append(f"{org}-{drug}: {val}")
        return {"stable": len(anomalies) == 0, "anomalies": anomalies}

    def get_antibiogram(self, organism: str) -> List[Dict]:
        """Returns a list of drugs and their current susceptibility for an organism."""
        org_data = self.susceptibility.get(organism, {})
        return [
            {"organism": organism, "drug": drug, "susceptibility_pct": round(val * 100, 1)}
            for drug, val in org_data.items()
        ]

# No global RESISTANCE_MODEL. Each environment instance should own its engine for isolation.
