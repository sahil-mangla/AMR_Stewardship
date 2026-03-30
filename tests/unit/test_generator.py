from env.generator import CaseGenerator
from env.resistance import ResistanceEngine

def test_patient_randomization():
    """Verify that the generator produces diverse patient data."""
    gen = CaseGenerator()
    eng = ResistanceEngine()
    case1 = gen.generate_case(eng)
    case2 = gen.generate_case(eng)
    assert case1["patient"].patient_id != case2["patient"].patient_id
    assert case1["patient"].age >= 18
    assert case1["patient"].weight_kg >= 50.0

def test_medical_error_injection():
    """Verify that the generator can produce clinical error scenarios."""
    gen = CaseGenerator()
    eng = ResistanceEngine()
    errors = set()
    for _ in range(100):
        case = gen.generate_case(eng)
        errors.add(case["context"]["error_type"])
    
    # We expect at least allergy, renal, and weight errors in a sample of 100
    assert "allergy" in errors
    assert "renal" in errors
    assert "weight" in errors
    assert "correct" in errors

def test_weight_based_logic():
    """Verify weight-adjust-needed metadata is set correctly."""
    gen = CaseGenerator()
    eng = ResistanceEngine()
    for _ in range(50):
        case = gen.generate_case(eng)
        if "vancomycin" in case["prescription"].drug.lower():
            assert case["context"]["weight_adjust_needed"] is True
