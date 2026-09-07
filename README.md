# SecureMesh-SCE

**AI-assisted Security Chaos Engineering testbed for studying adaptive attack and defence in IoT environments.**

SecureMesh-SCE extends the SecureMesh honeypot into a controlled, reproducible security-chaos environment. It formulates the interaction between an attacker and a defender as a Bayesian Markov game, and evaluates whether adaptive AI attackers and defenders change attack effectiveness and system resilience.

> **Research Question:** Can adaptive AI-based attackers discover and exploit security weaknesses in an IoT environment that are missed by deterministic attack strategies?

---

## Architecture

```
                     SecureMesh-SCE
                          |
            +-------------+-------------+
            |                           |
      AI ATTACKER                  AI DEFENDER
      (7 agent types)             (3 agent types)
            |                           |
       chooses action            chooses response
            |                           |
            +-------------+-------------+
                          |
                +---------+---------+
                |   IoT TESTBED     |
                |                   |
                | SSH Service       |
                | HTTP Service      |
                | ESP32 Sim (MQTT)  |
                | ESP8266 Sim       |
                | IDS Engine        |
                +---------+---------+
                          |
                     telemetry
                          |
                SCE Experiment Engine
                          |
              +-----------+-----------+
              |                       |
         Metrics               Experiment Record
         (ASR, DR, MTTD,      (JSON + CSV + JSONL)
          MTTR, Resilience)
              +-----------+-----------+
                          |
                   Matrix Runner
                   (Statistical analysis,
                    LaTeX / Markdown tables)
```

---

## Key Features

| Category | Details |
|---|---|
| **SCE Experiment Engine** | YAML-driven experiment lifecycle following the SCENE methodology (Jolak et al.). Hypothesis testing, time-stamped telemetry, structured JSON records. |
| **7 Attacker Types** | Random, Scripted (kill-chain), Aggressive, Stealthy, Recon-Heavy, RL (PPO), LLM-assisted (Gemini + Ollama). |
| **3 Defender Types** | Static (rule-based), ML (Random Forest), RL (PPO + Bayesian belief tracking). |
| **Experiment Matrix** | Run full attacker x defender grids with statistical testing (Mann-Whitney U, Cohen's d), LaTeX and Markdown table export. |
| **Realistic IoT Simulation** | ESP32/ESP8266 simulators with MQTT, CoAP, OTA attack surfaces. SSH and HTTP service simulators with IDS engine. |
| **Evaluation Framework** | Attack success rate, detection rate, MTTD, MTTR, false positive rate, system resilience, service availability. |

---

## Getting Started

### Prerequisites

- Python 3.10+
- (Optional) Ollama for local LLM attacker
- (Optional) Gemini API key for cloud LLM attacker

### Installation

```bash
pip install numpy gymnasium matplotlib pandas scikit-learn tabulate pyyaml
```

### Quick Start

```bash
# Run a single SCE experiment from YAML
python -m securemesh_testbed.experiments.engine \
    --config experiments/scenarios/02_ssh_bruteforce.yaml

# Run the original Markov game experiment
python -m securemesh_testbed.experiments.run_experiment \
    --scenario known_attacks --attacker scripted --defender rl \
    --episodes 50 --seed 42

# Run a full experiment matrix (3 attackers x 2 defenders x 2 seeds)
python -m securemesh_testbed.experiments.matrix_runner \
    --attackers scripted aggressive rl \
    --defenders static rl \
    --seeds 2 --episodes 50
```

---

## SCE Experiment Format

Experiments are defined in YAML:

```yaml
experiment:
  id: "SCE-002"
  name: "SSH Brute Force"
  target: "ssh"
  attacker: "aggressive"
  defender: "static"
  scenario: "known_attacks"
  duration: 80
  episodes: 50
  seed: 42

hypothesis:
  text: "The defender should block the SSH brute-force attack
         within 10 steps, keeping attack success rate below 30%."
  metric: "attack_success_rate"
  threshold: 0.3
  direction: "below"
```

Each experiment produces a structured JSON record:

```json
{
  "experiment_id": "SCE-002",
  "attacker_policy": "aggressive",
  "defender_policy": "static",
  "hypothesis": {"result": "PASS", "actual_value": 0.0427},
  "metrics": {
    "attack_success_rate": 0.0427,
    "detection_rate": 0.3457,
    "false_positive_rate": 0.1240,
    "attack_impact": 0.0831,
    "system_resilience": 0.8921,
    "mttd_steps": 10.15,
    "mttr_steps": 14.20
  }
}
```

---

## Research Questions

| RQ | Question |
|---|---|
| **RQ1** | Does an RL attacker achieve higher attack success than a deterministic attacker? |
| **RQ2** | Does adaptive defence reduce attack success and recovery time? |
| **RQ3** | Does repeated attacker-defender interaction improve system resilience? |

### Experiment Matrix

```
              Defender
            Static    ML    RL
Attacker
  Scripted    A       B     C
  Aggressive  D       E     F
  Stealthy    G       H     I
  ReconHeavy  J       K     L
  RL          M       N     O
  LLM         P       Q     R
```

---

## Project Structure

```
SecureMesh-SCE/
|
|-- securemesh_testbed/           Core Python package
|   |-- agents/
|   |   |-- attacker/             7 attacker implementations
|   |   |-- defender/             3 defender implementations
|   |   |-- common/              Shared PPO agent
|   |-- environment/              Network, service, IoT, IDS simulators
|   |-- game/                     Gymnasium Markov game
|   |-- experiments/              SCE engine, matrix runner, scenarios
|   |-- evaluation/               Metrics, logging, plotting, comparison
|   |-- config/                   Scenario configurations
|   |-- results/                  Experiment output
|
|-- experiments/
|   |-- scenarios/                YAML experiment definitions
|   |-- matrix_config.yaml        Matrix runner configuration
|
|-- docs/                         Architecture and methodology docs
|-- frontend/                     React dashboard (WIP)
|-- backend/                      FastAPI server
|-- esp8266/                      Hardware firmware
|-- docker-compose.yml            Cowrie + MongoDB stack
```

---

## LLM Attacker Configuration

Set environment variables to enable the LLM-assisted attacker:

```bash
# Gemini (cloud)
export GEMINI_API_KEY="your-api-key"
export LLM_PROVIDER="gemini"

# Ollama (local)
export LLM_PROVIDER="ollama"
export OLLAMA_MODEL="llama3.2"
```

The LLM attacker falls back to random action selection when no provider is available.

---

## License

This project is licensed under the [MIT License](LICENSE).
