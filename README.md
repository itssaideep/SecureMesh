# SecureMesh-SCE

**A SCENE-Guided, AI-Assisted Security Chaos Engineering Testbed for Studying Adaptive Cyberattack and Cyberdefence in IoT Environments.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests: 12 Passed](https://img.shields.io/badge/Tests-12%20Passed-brightgreen.svg)](tests/)

---

## 1. Research Overview

**SecureMesh-SCE** is a reproducible research platform engineered following the **SCENE guidelines** (Jolak et al., 2026) and the **TAFFAC threat assessment framework** (Sood et al., 2026). It investigates the foundational research question:

> **Core Research Question:** *How does uncertainty about attacker capabilities affect the effectiveness and resilience of adaptive AI-based defence in an IoT security-chaos environment?*

In real-world IoT environments, defenders face deep epistemic uncertainty: adversary capability, persistence, and strategic intent are hidden. Non-adaptive defences fail against novel kill-chains, while naive autonomous RL defenders often over-isolate the infrastructure, causing catastrophic self-inflicted outages.

SecureMesh-SCE models this dynamic interaction as a **two-player Bayesian Markov Game**, pitting an offense spectrum (from deterministic scripts to imitation and generative LLMs) against a defensive hierarchy that maintains an explicit Bayesian posterior belief distribution over latent attacker types, filtered through safety-constrained action gates.

---

## 2. Platform Architecture

```
                                  SecureMesh-SCE
                                         |
               +-------------------------+-------------------------+
               |                                                   |
      OFFENSE HIERARCHY                                   DEFENCE HIERARCHY
  (Levels 0–4 Policy Space)                           (Levels 1–5 Policy Space)
  - Level 0: Scripted (Deterministic)                 - Level 1: Static (Heuristic Rules)
  - Level 1: BC (Behavioral Cloning)                  - Level 2: ML (Random Forest)
  - Level 2: GAIL (Imitation Learning)                - Level 3: RL (PPO on System State)
  - Level 3: Adaptive PPO                             - Level 4: Bayesian RL (PPO + Belief)
  - Level 4: Generative LLM (Gemini/Ollama)           - Level 5: Constrained (Safety Gate)
               |                                                   |
               +-------------------------+-------------------------+
                                         |
                            BAYESIAN MARKOV GAME
                    [x_t: System | b_t: Belief | h_t: History | r_t: Risk]
                                         |
                     +-------------------+-------------------+
                     |                                       |
             SIMULATED TESTBED                       PHYSICAL MESH
           - Network Topology (DMZ, Corp, IoT)     - ESP8266 Sensor Nodes
           - SSH Simulator (Cowrie Semantics)      - ESP32 Gateway Node
           - HTTP/CGI Web Service                  - Local MQTT Broker
           - Mosquitto MQTT Broker                 - Docker Container Stack
                     |                                       |
                     +-------------------+-------------------+
                                         |
                             SCENE EXPERIMENT ENGINE
                      (8-Stage Chaos Experiment Lifecycle)
                                         |
               +-------------------------+-------------------------+
               |                                                   |
       EVALUATION ENGINE                                  DYNAMIC RISK REGISTER
  - Security (ASR, DR, F1)                           - TAFFAC Attribute Taxonomy
  - Response (MTTD, MTTR, Cost)                      - STRIDE Threat Categorization
  - Bayesian (Brier, ECE, Log Loss)                  - MITRE ATT&CK Technique Mapping
  - Resilience (Availability, Impact)                - Dynamic Residual Risk Tracking
  - Significance (Mann-Whitney U, Cohen's d)
```

---

## 3. Agent Hierarchies

### Offense Hierarchy (Levels 0–4)
1. **Level 0 — Scripted (`ScriptedAttacker`)**: Deterministic kill-chain execution (Recon $\to$ Auth $\to$ Exploit $\to$ Persist $\to$ Evasion).
2. **Level 1 — Behavioral Cloning (`BCAttacker`)**: Supervised policy trained on demonstration trajectories.
3. **Level 2 — GAIL (`GAILAttacker`)**: Generative Adversarial Imitation Learning matching adversary state-action occupancy distributions.
4. **Level 3 — Adaptive PPO (`PPOAttacker`)**: Autonomous reinforcement learning discovering defensive blind spots via clipped policy gradients.
5. **Level 4 — LLM-Assisted (`LLMAttacker`)**: Strategic prompt reasoning with Gemini 2.0 Flash or Ollama Llama 3.1:8b, with bounded JSON action schemas.

### Defence Hierarchy (Levels 1–5)
1. **Level 1 — Static (`StaticDefender`)**: Heuristic threshold baseline.
2. **Level 2 — Supervised ML (`RandomForestDefender`)**: Multi-class Random Forest predicting responses from telemetry features.
3. **Level 3 — RL Defender (`RLDefender`)**: PPO agent observing only physical system state $x_t$ without belief augmentation.
4. **Level 4 — Bayesian RL (`BayesianRLDefender`)**: PPO agent observing augmented state $s_t = [x_t, b_t, h_t, r_t]$, conditioning policy on posterior belief.
5. **Level 5 — Constrained Defender (`ConstrainedDefender`)**: Bayesian RL wrapped with an action safety gate enforcing staged escalation (`NORMAL` $\to$ `CONFIRMED` $\to$ `MITIGATE`) and blocking unauthorized critical interventions.

---

## 4. The 8-Stage SCENE Experiment Lifecycle

Every scenario executes according to the formal SCENE framework:
1. **Define Hypothesis**: Pre-registers expected quantitative threshold and direction (e.g., detection rate $\ge 0.60$).
2. **Establish Steady State**: Measures baseline stability (availability $\ge 0.99$, zero compromises).
3. **Select Scenario & Inject Chaos**: Injects adversary attacks or network chaos.
4. **Joint Action Execution**: Simultaneous attacker/defender transitions with reward calculation.
5. **Telemetry & Anomaly Detection**: Tracks packet drops, CPU load, and IDS alerts.
6. **Measure Outcome & Evaluate Hypothesis**: Generates statistical verdicts (`PASS` / `FAIL`) with 95% confidence intervals.
7. **Compute Residual Risk**: Dynamically updates control maturity and residual risk in the risk register.
8. **Export Structured Records**: Writes JSONL telemetry, CSV episode summaries, and LaTeX paper artifacts.

---

## 5. Canonical Scenarios (Experiments 1–8)

| Scenario | Config File | Attacker | Defender | Core Research Focus |
|:---:|:---|:---:|:---:|:---|
| **Exp 1** | `01_baseline.yaml` | Scripted | Static | Non-adaptive deterministic baseline |
| **Exp 2** | `02_defense_rl.yaml` | Scripted | RL | Un-augmented RL defence learning |
| **Exp 3** | `03_imitation.yaml` | BC | Static | Supervised imitation bypassing static rules |
| **Exp 4** | `04_attacker_adaptation.yaml` | PPO | Static | Adaptive adversary discovering blind spots |
| **Exp 5** | `05_co_adaptation.yaml` | PPO | RL | Dual-learning co-adaptation dynamics |
| **Exp 6** | `06_bayesian_defense.yaml` | PPO | Bayesian RL | Uncertainty advantage of Bayesian belief |
| **Exp 7** | `07_constrained_safety.yaml` | PPO | Constrained | Safety-gated bounded autonomy |
| **Exp 8** | `08_llm_offensive.yaml` | LLM | Bayesian RL | Generative LLM offensive planning |

---

## 6. Quick Start & Reproducibility

### Prerequisites & Installation
```bash
git clone https://github.com/itssaideep/SecureMesh.git
cd SecureMesh

# Install unified research dependencies
pip install -r requirements.txt
```

### Run Unit Test Suite
```bash
python -m pytest tests/ -v
```

### Run a Single SCENE Scenario
```bash
# Run Baseline (Experiment 1)
python -m securemesh_sce.experiments.runner --config experiments/scenarios/01_baseline.yaml --episodes 10

# Run Bayesian Defense (Experiment 6)
python -m securemesh_sce.experiments.runner --config experiments/scenarios/06_bayesian_defense.yaml --episodes 10

# Run Constrained Safety Gating (Experiment 7)
python -m securemesh_sce.experiments.runner --config experiments/scenarios/07_constrained_safety.yaml --episodes 10
```

### Run Full Multi-Seed Matrix & Ablation Suite
```bash
# Executes 5x5 Attacker x Defender matrix across multiple seeds
python scripts/run_all_ablations.py --episodes 5 --seeds 42 101 202 --duration 50
```

### Regenerate Research Paper Artifacts
```bash
python scripts/generate_paper_artifacts.py
```
Emits:
- `fig_belief_trajectory.png` / `.svg` (Bayesian posterior convergence)
- `fig_calibration_curve.png` / `.svg` (Reliability diagram with ECE)
- `fig_ablation_comparison.png` / `.svg` (Defensive component ablation)
- `tab_adversary_matrix.tex` (LaTeX booktabs cross-product table)
- `tab_ablation_study.tex` (LaTeX ablation table)
- `risk_register.md` (Updated residual risk register)

---

## 7. Research Documentation

- [Methodology & Game Formulation](docs/methodology.md)
- [TAFFAC Threat Model & MITRE ATT&CK](docs/threat_model.md)
- [Canonical Experiments & Ablations](docs/experiments.md)
- [Reproducibility & Execution Protocol](docs/reproducibility.md)
- [Threats to Validity & Limitations](docs/limitations.md)

---

## 8. License

This project is licensed under the [MIT License](LICENSE).
