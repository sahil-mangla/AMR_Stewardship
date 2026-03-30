# 🏥 Hospital ASP Coordinator — OpenEnv Environment

A real-world reinforcement learning environment simulating a **Hospital Antimicrobial Stewardship Program (ASP)** coordinator. An AI agent reviews antibiotic prescription requests and must make evidence-based clinical decisions under four competing objectives.

---

## Why This Domain?

Antimicrobial stewardship is one of the most impactful real-world decision-making tasks in medicine. Every year, antibiotic misuse drives antimicrobial resistance — a global health crisis responsible for over 1.2 million deaths annually. ASP coordinators must balance:

- **Patient safety** (prescribe the right drug, catch allergy conflicts)
- **Resistance prevention** (prefer narrow-spectrum agents, de-escalate)
- **Cost efficiency** (formulary-preferred drugs save hospitals $millions)
- **Prescriber satisfaction** (timely decisions, clear clinical communication)

This makes it a perfect RL environment: multi-objective, multi-step, with partial reward signals at every decision point.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Hospital ASP Coordinator Environment                   │
├─────────────────────────────────────────────────────────┤
│  External Tool APIs (simulated):                        │
│  ├── EHR System       → patient demographics, allergies │
│  ├── Lab System       → culture & sensitivity results   │
│  ├── Formulary DB     → drug availability & cost        │
│  ├── Resistance Tracker → hospital antibiogram          │
│  └── Communication    → prescriber messaging            │
│                                                         │
│  Multi-Step Workflow:                                   │
│  1. List pending prescription requests                  │
│  2. Retrieve patient data (allergies, renal function)   │
│  3. Check lab results (organism, sensitivities)         │
│  4. Check hospital antibiogram (resistance rates)       │
│  5. Make clinical decision (approve/modify/reject)      │
│  6. Communicate to prescriber                           │
│  7. Follow up on ICU/complex cases                      │
└─────────────────────────────────────────────────────────┘
```

---

## Action Space

| Action | Parameters | Description |
|---|---|---|
| `list_pending` | — | View prescription queue |
| `get_patient_data` | `patient_id` | EHR: demographics, allergies, renal function |
| `get_lab_results` | `patient_id` | Lab: culture organism + sensitivities |
| `check_formulary` | `drug` | Cost, availability, restrictions, alternatives |
| `check_antibiogram` | `organism` | Hospital resistance %s (last 12 months) |
| `approve_prescription` | `prescription_id` | Accept as written |
| `modify_prescription` | `prescription_id, drug, dose_mg, frequency, duration_days` | Change drug/dose/regimen |
| `reject_prescription` | `prescription_id, reason` | Reject with documented reason |
| `send_message` | `recipient_id, message, prescription_id` | Communicate with prescriber |
| `follow_up` | `patient_id` | Schedule clinical review |
| `noop` | — | Take no action |

---

## Observation Space

Each step returns a structured `Observation` containing:
- `pending_prescriptions` — list of Rx awaiting ASP review
- `patient_data` — populated after `get_patient_data` (allergies, eGFR, diagnosis, ICU status)
- `lab_results` — populated after `get_lab_results` (organism, sensitive_to, resistant_to)
- `formulary_info` — populated after `check_formulary`
- `antibiogram_data` — populated after `check_antibiogram`
- `last_action_result` — feedback from previous step
- `decisions_made` — history of clinical decisions this episode
- `messages_sent` — communication log
- `episode_reward_so_far` — running total

---

## Reward Function

The reward function is **continuous and multi-component** — not a sparse end-of-episode binary:

| Component | Weight | What Triggers It |
|---|---|---|
| `patient_safety` | 35% | Correct drug, allergy check, appropriate rejection |
| `resistance_stewardship` | 25% | Narrow-spectrum choice, justified broad-spectrum |
| `cost_efficiency` | 15% | Formulary-preferred low-cost agent |
| `prescriber_satisfaction` | 15% | Timely decisions, informative messages |
| `process_quality` | 10% | Gathering clinical data before deciding |
| `penalty` | additive | Allergy violation (−0.5), unsafe drug (−0.3), blind decision (−0.15), noop (−0.02) |

---

## Tasks

### Task 1 — EASY: Allergy Safety Check
- **Patient**: P001, 68F, documented **penicillin allergy**
- **Prescription**: Amoxicillin 500mg PO q8h × 7 days (pneumonia)
- **Goal**: Catch the allergy conflict, switch to safe alternative (azithromycin/doxycycline/ceftriaxone)
- **Baseline score**: ~1.0 for optimal agent, 0.0 if allergy missed

### Task 2 — MEDIUM: Resistance-Guided De-escalation
- **Patient**: P002, 34F, uncomplicated UTI
- **Prescription**: Ciprofloxacin 500mg PO q12h × 5 days
- **Challenge**: Hospital antibiogram shows only 62% E. coli susceptibility. Lab confirms resistance.
- **Goal**: Check antibiogram + lab, de-escalate to nitrofurantoin (94% susceptibility, first-line UTI)
- **Baseline score**: ~0.75–1.0 with proper tool use

### Task 3 — HARD: ICU Multi-Drug Stewardship
- **Patient**: P003, 55M, ICU, MRSA bacteremia, **eGFR = 22** (severe CKD), immunosuppressed
- **Allergies**: Sulfonamides + fluoroquinolones
- **Two prescriptions**: Vancomycin (standard dose — needs renal adjustment) + Meropenem (restricted drug)
- **Goal**: Adjust vancomycin to q48h/reduced dose for renal impairment, approve meropenem with documented justification, communicate to ICU team, schedule follow-up
- **Baseline score**: ~0.6–0.8 depending on completeness

---

## Setup & Usage

### Local Development

```bash
# Clone repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/hospital-asp-coordinator
cd hospital-asp-coordinator

# Install dependencies
pip install -r requirements.txt

# Run the environment server
python server.py
# Server starts on http://localhost:7860

# Test the API
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"task_1"}'
```

### Docker

```bash
docker build -t asp-env .
docker run -p 7860:7860 asp-env

# Run inference (requires HF_TOKEN + MODEL_NAME)
export HF_TOKEN="hf_..."
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

### Running the Baseline Inference Script

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="hf_your_token_here"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

Output:
```
============================================================
  BASELINE SCORES SUMMARY
============================================================
  task_1 (easy  ): 0.7500
  task_2 (medium): 0.7500
  task_3 (hard  ): 0.5000
  Average        : 0.6667
============================================================
```

---

## API Reference

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/health` | GET | — | `{"status":"ok"}` |
| `/tasks` | GET | — | Task list with difficulty |
| `/reset` | POST | `{"task_id":"task_1"}` | `{observation}` |
| `/step` | POST | `{"task_id":..., "action_type":..., "parameters":{...}}` | `{observation, reward, done, info}` |
| `/state` | GET | `?task_id=task_1` | Full state snapshot |
| `/grade` | POST | `{"task_id":"task_1"}` | `{score: 0.0–1.0}` |

---

## OpenEnv Spec Compliance

- ✅ `reset()` → returns clean initial `Observation`
- ✅ `step(action)` → returns `(Observation, reward, done, info)`
- ✅ `state()` → returns full serialisable state dict
- ✅ Typed Pydantic models for Observation, Action, Reward
- ✅ `openenv.yaml` with full metadata
- ✅ 3+ tasks with agent graders (easy → medium → hard)
- ✅ Reward signals over full trajectory (not just end-of-episode)
- ✅ Deterministic, reproducible graders
- ✅ Baseline inference script (`inference.py`)
- ✅ Dockerfile with `docker build && docker run`
- ✅ HF Spaces deployment on port 7860

---

## Project Structure

```
.
├── inference.py          # Baseline inference script (competition requirement)
├── server.py             # FastAPI HTTP server
├── openenv.yaml          # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
├── README.md
├── env/
│   ├── __init__.py
│   ├── models.py         # Typed Observation, Action, Reward models
│   ├── asp_env.py        # Main ASPEnv class (reset/step/state)
│   └── reward.py         # Multi-objective reward function
├── tasks/
│   ├── __init__.py
│   └── tasks.py          # 3 tasks with deterministic graders
├── tools/
│   ├── __init__.py
│   └── tools.py          # EHR, Lab, Formulary, Antibiogram, Communication APIs
└── graders/
    └── __init__.py
```
