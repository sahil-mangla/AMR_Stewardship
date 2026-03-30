import pytest
from env.npc import PrescriberNPC

def test_npc_reputation_gain():
    """Verify reputation increases with good stewardship."""
    npc = PrescriberNPC("Dr. Test", "Internal Medicine")
    npc.personality = "cooperative"
    base_rep = npc.reputation
    npc.generate_response("This is a stewardship de-escalation.", "modify_prescription", "amoxicillin")
    assert npc.reputation > base_rep

def test_conservative_doctor_resistance():
    """Verify conservative doctors resist de-escalation at low reputation."""
    npc = PrescriberNPC("Dr. No", "Internal Medicine")
    npc.personality = "conservative"
    npc.reputation = 0.5
    
    response = npc.generate_response("We should narrow the therapy.", "modify_prescription", "amoxicillin")
    assert "hesitant to de-escalate" in response

def test_defensive_doctor_pushback():
    """Verify defensive doctors push back if not justified."""
    npc = PrescriberNPC("Dr. Defense", "Internal Medicine")
    npc.personality = "defensive"
    
    response = npc.generate_response("Switch this to vancomycin.", "modify_prescription", "vancomycin")
    assert "explain the clinical benefit" in response
    assert npc.satisfaction < 0.5
