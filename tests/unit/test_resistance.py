import pytest
from env.resistance import ResistanceEngine

def test_evolution_over_usage():
    """Verify that broad-spectrum usage decreases susceptibility."""
    engine = ResistanceEngine()
    base = engine.get_susceptibility("Escherichia coli", "meropenem")
    
    # Record multiple uses
    for _ in range(10):
        engine.record_usage("meropenem", is_broad_spectrum=True)
    
    new_val = engine.get_susceptibility("Escherichia coli", "meropenem")
    assert new_val < base

def test_outbreak_scenario():
    """Verify that outbreaks cause immediate drops to biological minimum."""
    engine = ResistanceEngine()
    engine.trigger_outbreak("Klebsiella pneumoniae", "meropenem")
    
    val = engine.get_susceptibility("Klebsiella pneumoniae", "meropenem")
    assert val == 0.05

def test_stability_validation():
    """Verify the stability monitor catches anomalies."""
    engine = ResistanceEngine()
    engine.susceptibility["Escherichia coli"]["meropenem"] = 1.2
    
    stability = engine.validate_stability()
    assert stability["stable"] is False
    assert "Escherichia coli-meropenem" in stability["anomalies"][0]
