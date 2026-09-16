# Reproducibility and Execution Guide

This document provides step-by-step instructions for independently reproducing all experimental findings, statistical evaluations, LaTeX tables, and plots in **SecureMesh-SCE**.

---

## 1. Environment Setup

### System Prerequisites
- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows (PowerShell/Bash)
- **Virtual Environment**: Recommended

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install unified research dependencies
pip install -r requirements.txt
```

---

## 2. Automated Test Suite Execution

Validate that all mathematical transitions, exact Bayesian inference routines, PPO agent forward/backward passes, and scenario parsers function correctly:

```bash
python -m pytest tests/ -v
```
All 12 unit tests should pass with green status.

---

## 3. Running Single Canonical Scenarios

Execute any of the eight canonical SCENE scenarios using the CLI orchestrator:

```bash
# Run Scenario 1: Baseline (Scripted vs Static)
python -m securemesh_sce.experiments.runner --config experiments/scenarios/01_baseline.yaml --episodes 10

# Run Scenario 6: Bayesian Belief Advantage (PPO vs Bayesian RL)
python -m securemesh_sce.experiments.runner --config experiments/scenarios/06_bayesian_defense.yaml --episodes 10

# Run Scenario 7: Bounded Autonomy & Safety Gating
python -m securemesh_sce.experiments.runner --config experiments/scenarios/07_constrained_safety.yaml --episodes 10
```

Each run generates:
- `<experiment_id>_telemetry.jsonl`: Real-time per-step observation records.
- `<experiment_id>_episodes.csv`: Per-episode performance metrics.
- `<experiment_id>_summary.json`: Provenance metadata, 95% confidence intervals, and hypothesis verdict.

---

## 4. Running the Complete Multi-Seed Ablation Suite

To execute the entire $5 \times 5$ Attacker-Defender matrix across multiple random seeds:

```bash
python scripts/run_all_ablations.py --episodes 5 --seeds 42 101 202 --duration 50
```

---

## 5. Regenerating Paper Artifacts

To regenerate all LaTeX booktabs tables (`tab_adversary_matrix.tex`, `tab_ablation_study.tex`), Markdown tables, and high-resolution vector figures (`.png` and `.svg`):

```bash
python scripts/generate_paper_artifacts.py
```

All artifacts will be placed in `experiments/results/` ready for publication.
