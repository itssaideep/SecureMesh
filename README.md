# SecureMesh: Game-Theoretic RL for Adaptive Cyberattack & Cyberdefence

**A Game-Theoretic Reinforcement Learning Testbed for Studying Co-Evolutionary AI Cyberattack and Cyberdefence in IoT Environments.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Hardware: ESP8266](https://img.shields.io/badge/Hardware-NodeMCU%20ESP8266-red.svg)](esp8266/)
[![Tests: Passing](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)

---

## 1. Research Overview

Modern software systems and interconnected IoT deployments face an evolving threat landscape driven by **offensive AI** — autonomous, adaptive cyberattack agents capable of dynamically shifting tactics. Traditional defence mechanisms relying on static heuristics or rule-based firewalls struggle to keep pace with adversaries that alter their decision logic in response to defensive interventions.

This project implements a game-theoretic reinforcement learning framework designed to study:
1. **Attack Effectiveness:** How autonomous AI attackers learn optimal exploit paths against protected systems.
2. **Defensive Adaptation:** How AI defenders learn to detect and mitigate evolving attacks while preserving system availability.
3. **Adversarial Co-Evolution:** The strategic dynamics, policy cycling, and equilibrium behavior when both attacker and defender learn simultaneously via reinforcement learning in a shared target environment.

---

## 2. Theoretical Architecture

The interaction between the attacker and defender is formulated as a **two-player simultaneous-action Markov Game**:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}_{\text{atk}}, \mathcal{A}_{\text{def}}, \mathcal{P}, \mathcal{R}_{\text{atk}}, \mathcal{R}_{\text{def}}, \gamma \rangle$$

```
                          Two-Player Markov Game
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
      AI Attacker                                       AI Defender
  ┌─────────────────┐                               ┌─────────────────┐
  │ - Scripted (L0) │                               │ - Static (D1)   │
  │ - PPO (L1)      │                               │ - RL (D2 / PPO) │
  └────────┬────────┘                               └────────┬────────┘
           │ Action a_atk                                    │ Action a_def
           ▼                                                 ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                   Target Environment                         │
   │  - Network Topology (DMZ, Corporate, IoT Subnets)            │
   │  - Simulated Services (SSH, HTTP)                            │
   │  - Physical Hardware Honeypot (ESP8266)                      │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
               Joint Transition & Multi-Objective Rewards
               - Attacker Reward: Exploitation success - Cost
               - Defender Reward: Detection + Availability - Disruption cost
```

### Action Spaces
- **Attacker Actions (13 discrete actions):**
  `NOOP`, `RECON_SCAN`, `RECON_FINGERPRINT`, `AUTH_BRUTEFORCE`, `AUTH_CREDENTIAL_STUFF`, `EXPLOIT_SERVICE`, `EXPLOIT_IOT`, `MALWARE_DROP`, `PERSIST_BACKDOOR`, `PERSIST_C2`, `EVADE_OBFUSCATE`, `EVADE_SLOWDOWN`, `LATERAL_MOVE`.
- **Defender Actions (14 discrete actions across 4 impact tiers):**
  - *Tier 0 (Monitor):* `NOOP`, `MONITOR`, `ANALYZE_LOGS`
  - *Tier 1 (Low Impact):* `ALERT_OPS`, `RATE_LIMIT_IP`, `ROTATE_CREDENTIALS`
  - *Tier 2 (Medium Impact):* `BLOCK_IP`, `DEPLOY_HONEYPOT`, `RESTART_SERVICE`, `ROLLBACK_CONFIG`
  - *Tier 3 (High / Critical Impact):* `ISOLATE_HOST`, `PATCH_VULNERABILITY`, `RELOAD_FIRMWARE`, `NETWORK_SHUTDOWN`

### Multi-Objective Reward Formulation
- **Attacker Reward:** Rewards successful compromise of services/hosts, penalizes detection and execution costs.
- **Defender Reward:** Rewards correct detections and swift mitigations, penalizes false alarms, successful adversary intrusions, and self-inflicted service downtime (preventing over-aggressive shutdowns).

---

## 3. Directory Structure

```
honeypot/
├── backend/                      # FastAPI REST & WebSocket server
│   ├── app.py                    # Entrypoint & route mounting
│   └── routes/                   # API routes (experiment runner, telemetry streaming)
├── esp8266/                      # Physical IoT honeypot firmware
│   ├── esp8266.ino               # Multi-service honeypot firmware
│   ├── platformio.ini            # PlatformIO build configuration
│   └── secrets.h.example         # Template for WiFi & backend credentials
├── experiments/                  # Experiment configurations and outputs
│   ├── matrix_config.yaml        # 2×2 experiment matrix specification
│   ├── results/                  # Generated telemetry, logs, and summaries
│   └── scenarios/                # Repeatable scenario definitions
│       ├── 01_baseline.yaml      # Scripted Attacker vs Static Defender
│       ├── 02_rl_attacker.yaml   # PPO Attacker vs Static Defender
│       ├── 03_rl_defender.yaml   # Scripted Attacker vs RL Defender
│       └── 04_co_evolution.yaml  # PPO Attacker vs RL Defender (Co-Evolution)
├── securemesh_sce/               # Core research package
│   ├── agents/                   # Agent hierarchies
│   │   ├── attacker/             # ScriptedAttacker, PPOAttacker
│   │   └── defender/             # StaticDefender, RLDefender
│   ├── environment/              # Target system simulation & hardware bridge
│   │   ├── iot/                  # ESP8266 simulator and physical bridge
│   │   ├── network/              # Topology and host models
│   │   ├── services/             # SSH and HTTP simulated services
│   │   └── telemetry/            # State observation and anomaly detection
│   ├── evaluation/               # Metrics and statistical tests
│   │   ├── metrics.py            # ASR, DR, Availability, MTTD, MTTR, Entropy
│   │   ├── plots.py              # Publication-grade plotting
│   │   └── statistics.py         # Hypothesis testing & confidence intervals
│   ├── experiments/              # Execution engine and CLI runner
│   │   ├── engine.py             # Multi-episode experiment lifecycle
│   │   └── runner.py             # CLI scenario runner
│   ├── game/                     # Two-player Markov game core
│   │   ├── actions.py            # Action definitions and impact tiers
│   │   ├── bayesian_game.py      # Gymnasium environment
│   │   ├── rewards.py            # Multi-objective reward functions
│   │   ├── state.py              # State representations
│   │   └── transitions.py        # Transition engine
│   └── logging/                  # Experiment and telemetry loggers
├── tests/                        # Pytest suite
│   ├── test_agents.py            # Unit tests for attacker and defender agents
│   ├── test_game.py              # Environment dynamics and reward tests
│   └── test_scene_engine.py      # Experiment engine and hypothesis tests
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation
```

---

## 4. Scenarios & Research Questions

| Scenario | Attacker | Defender | Research Focus | Key Hypothesis |
|:---|:---:|:---:|:---|:---|
| **`01_baseline.yaml`** | Scripted | Static | Benchmark performance of fixed policies | Static rules detect predictable kill-chains ($\text{DR} \ge 50\%$) |
| **`02_rl_attacker.yaml`** | PPO | Static | Offensive AI adaptation against static rules | RL attacker discovers evasion paths ($\text{ASR} \ge 40\%$) |
| **`03_rl_defender.yaml`** | Scripted | RL | Defensive AI learning against known patterns | RL defender mitigates attacks ($\text{DR} \ge 65\%$, $\text{Avail} \ge 80\%$) |
| **`04_co_evolution.yaml`** | **PPO** | **RL** | **Simultaneous adversarial co-evolution** | Defender preserves system resilience ($\text{Avail} \ge 70\%$) |

---

## 5. Quick Start & Execution

### Installation

```bash
# Clone the repository
git clone https://github.com/itssaideep/SecureMesh.git
cd SecureMesh

# Set up Python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

Execute any of the 4 thesis scenarios from the command line:

```bash
# 1. Deterministic baseline
python -m securemesh_sce.experiments.runner --config experiments/scenarios/01_baseline.yaml --episodes 10

# 2. Adaptive RL Attacker
python -m securemesh_sce.experiments.runner --config experiments/scenarios/02_rl_attacker.yaml --episodes 10

# 3. Adaptive RL Defender
python -m securemesh_sce.experiments.runner --config experiments/scenarios/03_rl_defender.yaml --episodes 10

# 4. Adversarial Co-Evolution (Core Experiment)
python -m securemesh_sce.experiments.runner --config experiments/scenarios/04_co_evolution.yaml --episodes 10
```

### Running the Live Backend Server

```bash
# Start FastAPI backend
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

- API Docs: `http://localhost:8000/docs`
- WebSocket Live Telemetry: `ws://localhost:8000/api/experiment/ws`

### Running Unit Tests

```bash
python -m pytest tests/ -v
```

---

## 6. Physical Hardware Honeypot (ESP8266)

The framework includes firmware for a physical **NodeMCU ESP8266** acting as an IoT honeypot endpoint:
- **Port 80 HTTP:** Traps suspicious HTTP request paths (`/admin`, `/wp-login.php`, `/cgi-bin/`).
- **Port 23 Telnet:** Traps brute-force credential attempts.
- **Port 8080 Management:** Traps unauthenticated backdoor access.
- **Telemetry Bridge:** Events captured on the physical ESP8266 are forwarded via WiFi to the FastAPI backend and integrated into the experiment state via `PhysicalESP8266Bridge`.

### Hardware Configuration
1. Copy `esp8266/secrets.h.example` to `esp8266/secrets.h`.
2. Configure your WiFi credentials and backend IP address:
   ```cpp
   #define WIFI_SSID "YourNetworkName"
   #define WIFI_PASSWORD "YourNetworkPassword"
   #define BACKEND_HOST "192.168.1.100"
   #define BACKEND_PORT 8000
   ```
3. Compile and flash using PlatformIO:
   ```bash
   pio run -d esp8266 -t upload
   ```

---

## 7. License & Citation

This project is licensed under the MIT License.
