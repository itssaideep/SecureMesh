# SecureMesh: Game-Theoretic Reinforcement Learning for Adaptive Cyberattack & Cyberdefence

**An Open-Source Game-Theoretic Testbed for Investigating Offensive AI Adaptation, Defensive AI Mitigation, and Adversarial Co-Evolution in IoT Environments.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-green.svg)](https://gymnasium.farama.org/)
[![LLM: Gemini 3.5 Flash](https://img.shields.io/badge/LLM-Gemini%203.5%20Flash-orange.svg)](experiments/scenarios/05_llm_vs_llm.yaml)
[![Hardware: ESP8266](https://img.shields.io/badge/Hardware-NodeMCU%20ESP8266-red.svg)](esp8266/)
[![Tests: 10 Passed](https://img.shields.io/badge/Tests-10%20Passed-brightgreen.svg)](tests/)

---

## Table of Contents

1. [Thesis Motivation & Research Questions](#1-thesis-motivation--research-questions)
2. [Game-Theoretic Formulation](#2-game-theoretic-formulation)
3. [Agent Architectures](#3-agent-architectures)
4. [Action Spaces & Impact Tiers](#4-action-spaces--impact-tiers)
5. [Evaluation Metrics & Statistical Analysis](#5-evaluation-metrics--statistical-analysis)
6. [Experimental Scenarios](#6-experimental-scenarios)
7. [Target Environment & Physical Honeypot](#7-target-environment--physical-honeypot)
8. [Repository Architecture](#8-repository-architecture)
9. [Installation & Quick Start](#9-installation--quick-start)
10. [REST & WebSocket API Reference](#10-rest--websocket-api-reference)
11. [License & Citation](#11-license--citation)

---

## 1. Thesis Motivation & Research Questions

Recent advances in artificial intelligence have significantly expanded the cyber threat surface. **Offensive AI** — adversaries empowered by reinforcement learning and autonomous decision logic — can dynamically probe target systems, evade static signature-based detection rules, and exploit interconnected services. In response, defensive systems must not only detect anomalous traffic but also learn when and how to intervene without causing self-inflicted service downtime (over-aggressive containment).

This project investigates five core research questions across game-theoretic reinforcement learning and generative AI:

| Research Question | Scientific Objective | Target Scenario | Primary Metrics |
|:---|:---|:---:|:---|
| **RQ1: Offensive AI Adaptation** | Can an autonomous RL attacker systematically discover evasion paths and exploit vulnerabilities against fixed heuristic defenses? | `02_rl_attacker.yaml` | Attack Success Rate (ASR), Action Entropy |
| **RQ2: Defensive AI Learning** | Can an autonomous RL defender learn to detect and mitigate multi-stage attack kill-chains while preserving critical service availability? | `03_rl_defender.yaml` | Detection Rate (DR), Service Availability, MTTR |
| **RQ3: Adversarial Co-Evolution** | How do attacker and defender policies co-evolve when both learn concurrently in a two-player Markov game? Does the system converge to an equilibrium or exhibit strategic cycling? | `04_co_evolution.yaml` | Reward Trajectories, Pareto Efficiency, Availability |
| **RQ4: Cyber-Physical Grounding** | How do real-world embedded hardware constraints (memory saturation, connection dropouts, network jitter) translate into game-theoretic dynamics? | Physical ESP8266 Bridge | Telemetry latency, Service Availability, Free Heap |
| **RQ5: Generative Adversaries on IoT (Frontier)** | Can Large Language Models (Google Gemini 3.5 Flash) reason over MITRE tactics to plan multi-step kill-chains and execute defensive SOC triage on physical IoT endpoints? | `05_llm_vs_llm.yaml` | Detection Rate (DR), Availability, Strategic Reasoning |

---

## 2. Game-Theoretic Formulation

The interaction is formalized as a **two-player simultaneous-action Markov Game**:

```text
M = ⟨ S, A_atk, A_def, P, R_atk, R_def, γ ⟩
```

```
                           Simultaneous Markov Game
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
      AI Attacker                                         AI Defender
  ┌──────────────────────┐                           ┌─────────────────────┐
  │ - Scripted (L0)      │                           │ - Static Rules (D1) │
  │ - PPO Policy (L1)    │                           │ - PPO Policy (D2)   │
  │ - LLM Agent (L2)     │                           │ - LLM SOC (D3)      │
  └──────────┬───────────┘                           └──────────┬──────────┘
             │ Action a_atk ∈ A_atk                             │ Action a_def ∈ A_def
             ▼                                                  ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                       Target Environment                          │
   │  - Subnet Topology (DMZ, Corporate LAN, IoT Subnet)               │
   │  - Simulated Vulnerable Services (SSH, HTTP)                      │
   │  - Physical Hardware Honeypot Endpoint (NodeMCU ESP8266)          │
   └─────────────────────────────────┬─────────────────────────────────┘
                                     │
              Joint Transition P(s_{t+1} | s_t, a_atk, a_def)
              Multi-Objective Reward Vector (R_atk, R_def)
```

### State Space Representation
At time step `t`, the system state `s_t` observed by the agents consists of:

```text
s_t = [ x_t,  h_t,  r_t ]
```

- **Network State Vector (`x_t`):** Host isolation flags, compromised service states, active connection counts, port vulnerabilities, and IoT device firmware integrity.
- **Attack History Vector (`h_t`):** Sliding-window frequencies of reconnaissance scans, authentication failures, exploit attempts, and lateral movement.
- **Risk & Resilience Vector (`r_t`):** Cumulative service impact, current service availability (0.0 to 1.0), and active security escalation stage (`NORMAL`, `SUSPICIOUS`, `CONFIRMED`, `RECOVER`).

### Multi-Objective Reward Functions

#### Attacker Reward Function:
```text
R_atk = (w_comp × 𝟙_compromise) - (w_det × 𝟙_detected) - cost(a_atk)
```
- **Rewards:** Gaining unauthorized control over services and IoT nodes (`+w_comp`).
- **Penalties:** Triggering defensive intrusion alarms (`-w_det`) and action execution overhead (`-cost`).

#### Defender Reward Function:
```text
R_def = (w_det × 𝟙_correct_detection) + (w_avail × Availability) 
        - (w_fp × 𝟙_false_alarm) - (w_dos × 𝟙_unnecessary_shutdown) 
        + [(service_availability - 1.0) × 50.0] - cost(a_def)
```
- **Rewards:** Timely detection (`+w_det`) and maintaining service availability (`+w_avail × Availability`).
- **Continuous Penalty:** Linear availability gradient `(service_availability - 1.0) × 50.0` preventing the defender from collapsing into an all-shutdown local optimum.
- **Penalties:** False alarms (`-w_fp`), action execution overhead (`-cost`), and unnecessary shutdowns / self-inflicted Denial of Service (`-w_dos`).


---

## 3. Agent Architectures

The testbed features three attacker strategies and three defender strategies, structured into a **Core 2 × 2 Game-Theoretic RL Evaluation** plus an **Advanced Generative Frontier Duel**:

```
                            Offense vs Defence Evaluation Matrix
                      ┌───────────────────┬───────────────────┬───────────────────┐
                      │ Static Defender   │ RL Defender (PPO) │ LLM Defender      │
┌─────────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Scripted Attacker   │ SCE-001: Baseline │ SCE-003: Def-RL   │ —                 │
├─────────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ PPO Attacker        │ SCE-002: Atk-RL   │ SCE-004: Co-Evol  │ —                 │
├─────────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ LLM Attacker        │ —                 │ —                 │ SCE-005: Gen-Duel │
└─────────────────────┴───────────────────┴───────────────────┴───────────────────┘
```

### Attacker Agents
1. **`ScriptedAttacker` (Level 0 Baseline):**
   - Follows deterministic kill-chain stages: Reconnaissance $\to$ Weaponisation $\to$ Credential Brute-Force $\to$ Service Exploitation $\to$ Malware Delivery $\to$ Persistence $\to$ Lateral Movement.
   - Provides an empirical control baseline to measure standard defense performance.
2. **`PPOAttacker` (Level 1 Adaptive AI):**
   - Implements Proximal Policy Optimization (PPO) with clipped surrogate objective.
   - State input: Observed system state vector $x_t$.
   - Action output: Categorical probability distribution over 13 attacker actions.
   - Dynamically adapts its strategy to discover vulnerabilities and evade defensive detection.
3. **`LLMAttacker` (Level 2 Generative AI Adversary):**
   - Leverages **Google Gemini 3.5 Flash** (with automatic cascading failover to local **Ollama** `llama3.1:8b`).
   - Receives system state, network posture, and historical action outcomes formatted as structured context.
   - Generates tactical reasoning and selects the optimal next attack step mapped to MITRE ATT&CK techniques with JSON enforcement.

### Defender Agents
1. **`StaticDefender` (Level 1 Baseline):**
   - Evaluates rule-based heuristic thresholds on telemetry:
     - High scan rates trigger IP rate limiting.
     - Authentication failures trigger IP block rules.
     - Confirmed compromise triggers service restarts.
   - Represents traditional signature/heuristic intrusion response systems.
2. **`RLDefender` (Level 2 Adaptive AI):**
   - Implements PPO actor-critic network trained to optimize multi-objective security resilience.
   - State input: Concatenated state vector `[x_t, h_t, r_t]`.
   - Action output: Probability distribution over 14 tiered defender actions.
   - Learns proportional response: balances intrusion containment against service availability via continuous reward gradients.
3. **`LLMDefender` (Level 3 Generative SOC Analyst):**
   - Autonomous blue-team agent powered by **Google Gemini 3.5 Flash** (with local Ollama fallback).
   - Ingests real-time security alerts, intrusion detection flags, and service health metrics.
   - Synthesizes risk assessment and applies calibrated containment actions (from monitor to isolate) while protecting SLA uptime.

### Unified Dual-Backend `LLMClient`
Both LLM agents utilize a robust, shared `LLMClient` (`securemesh_sce/agents/common/llm_client.py`) featuring:
- **Primary Provider:** Google Gemini API (`gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-flash-latest`) via direct REST with `x-goog-api-key` authentication.
- **Secondary Provider:** Local headless Ollama server (`http://localhost:11434/api/generate`) with `llama3.1:8b` or `qwen3:8b`.
- **Automatic Fallback:** Gracefully cascades to local models upon HTTP 429 (rate limit) or 503 (service unavailable) responses.
- **Strict JSON Parsing:** Enforces deterministic JSON outputs with automatic markdown code-block extraction and schema validation.

---

## 4. Action Spaces & Impact Tiers

### Attacker Action Space (13 Actions)
| Action Enum | Description | MITRE ATT&CK Mapping | Cost |
|:---|:---|:---|:---:|
| `NOOP` | Idle step (maintain stealth) | T1027 (Obfuscation) | 0.0 |
| `RECON_SCAN` | Port and service discovery | T1046 (Network Service Discovery) | 0.5 |
| `RECON_FINGERPRINT`| OS and banner fingerprinting | T1082 (System Information Discovery)| 0.8 |
| `AUTH_BRUTEFORCE` | Dictionary password attack | T1110.001 (Password Guessing) | 1.5 |
| `AUTH_CREDENTIAL_STUFF` | Stolen credential stuffing | T1110.004 (Credential Stuffing) | 1.2 |
| `EXPLOIT_SERVICE` | Remote code execution on HTTP/SSH| T1190 (Exploit Public-Facing App) | 3.0 |
| `EXPLOIT_IOT` | Embedded device memory exploit | T1210 (Exploitation of Remote Services)| 3.5 |
| `MALWARE_DROP` | Payload delivery to host | T1204 (User Execution: Malicious File)| 2.5 |
| `PERSIST_BACKDOOR` | Webshell / listener installation | T1505 (Server Software Component) | 2.0 |
| `PERSIST_C2` | Beaconing connection setup | T1071 (Application Layer Protocol) | 2.5 |
| `EVADE_OBFUSCATE` | Traffic encryption & packet padding| T1027 (Obfuscated Information) | 1.0 |
| `EVADE_SLOWDOWN` | Low-and-slow rate-limited probes | T1036 (Masquerading) | 0.5 |
| `LATERAL_MOVE` | Pivot from DMZ to internal host | T1021 (Remote Services) | 3.0 |

### Defender Action Space (14 Actions across 4 Impact Tiers)
| Impact Tier | Action Enum | Action Description | Availability Impact | Execution Cost |
|:---|:---|:---|:---:|:---:|
| **Tier 0: Low (Autonomous)** | `NOOP` | Maintain current posture | None | 0.0 |
| | `MONITOR` | Passive sensor collection | None | 0.0 |
| | `INCREASE_MONITORING` | Elevate sensor polling frequency | None | 1.0 |
| | `ADJUST_IDS_THRESHOLD` | Tune detection sensitivity threshold | None | 2.0 |
| **Tier 1: Medium (Guarded)** | `HONEYPOT_DEPLOY` | Deploy or redirect to decoy trap | Low | 3.0 |
| | `RATE_LIMIT` | Throttle suspicious source address | Low | 5.0 |
| | `TERMINATE_SESSION` | Sever active suspicious TCP/SSH session | Low | 8.0 |
| | `RESET_CREDENTIALS` | Invalidate and rotate service credentials | Low | 10.0 |
| **Tier 2: High (Approval)** | `PATCH_SERVICE` | Hot-patch identified vulnerability | Medium | 12.0 |
| | `BLOCK_IP` | Drop offending IP at firewall | Low | 15.0 |
| | `ISOLATE_IOT` | Disconnect IoT endpoint from network | High | 20.0 |
| | `ISOLATE_SERVICE` | Quarantine compromised daemon | High | 25.0 |
| **Tier 3: Critical (Restricted)** | `RESTORE_SERVICE` | Re-enable and restore quarantined service | Recovery | 8.0 |
| | `NETWORK_SHUTDOWN` | Full subnet isolation (Emergency kill-switch) | Critical (100% loss) | 50.0 |

---

## 5. Evaluation Metrics & Statistical Analysis

Experiments automatically record step-level and episode-level metrics:

### 1. Security Containment
- **Attack Success Rate (ASR):** `ASR = Successful Attacks / Total Attack Actions`. Measures attacker exploit efficacy.
- **Detection Rate (DR):** `DR = True Positives / Total Attack Actions`. Measures defender IDS coverage.
- **Precision, Recall, F1-Score, False Positive Rate (FPR):** Standard classification performance of defensive alerts.

### 2. Operational Resilience
- **Mean Service Availability:** `Availability = (1 / T) × ∑ Availability_t` (ranging from 0.0 to 1.0). Measures average uptime of critical services during attack campaigns.
- **Mean Time to Detect (MTTD):** Elapsed steps between first attacker intrusion attempt and first true positive detection.
- **Mean Time to Respond (MTTR):** Elapsed steps between detection and decisive mitigation action (Tier 2 or Tier 3).
- **Cumulative Impact Score:** Aggregated damage score incurred across hosts and IoT nodes.

### 3. Adaptation Dynamics
- **Action Entropy:** `H(A) = -∑ P(a) × log2(P(a))`. Quantifies policy diversity and stealth unpredictability.
- **Statistical Hypothesis Evaluation:** Automatic computation of empirical mean, 95% Bootstrap Confidence Intervals (CI), and two-sample Mann-Whitney U tests against configured research thresholds.

---

## 6. Experimental Scenarios

All experiments are defined declaratively in `experiments/scenarios/`:

```
experiments/scenarios/
├── 01_baseline.yaml      # Scripted Attacker vs Static Defender (Deterministic Baseline)
├── 02_rl_attacker.yaml   # PPO Attacker vs Static Defender (Offensive AI Adaptation)
├── 03_rl_defender.yaml   # Scripted Attacker vs RL Defender (Defensive AI Learning)
├── 04_co_evolution.yaml  # PPO Attacker vs RL Defender (Adversarial Co-Evolution)
└── 05_llm_vs_llm.yaml    # LLM Attacker vs LLM Defender (Generative Red vs Blue Duel)
```

### Scenario Specifications

#### Scenario 1: `01_baseline.yaml` (Control Group)
- **Attacker:** `ScriptedAttacker` | **Defender:** `StaticDefender`
- **Objective:** Benchmarks system performance under non-adaptive, deterministic conditions.
- **Hypothesis:** The static defender will detect predictable scripted kill-chains with at least 50% detection rate without compromising service availability.

#### Scenario 2: `02_rl_attacker.yaml` (Offensive Adaptation)
- **Attacker:** `PPOAttacker` | **Defender:** `StaticDefender`
- **Objective:** Evaluates whether autonomous offensive AI learns to circumvent fixed defensive heuristics.
- **Hypothesis:** The adaptive PPO attacker will achieve an attack success rate exceeding 40% against the static defender.

#### Scenario 3: `03_rl_defender.yaml` (Defensive Learning)
- **Attacker:** `ScriptedAttacker` | **Defender:** `RLDefender`
- **Objective:** Evaluates whether an RL defender learns optimal containment strategies against known attacks without falling into self-inflicted denial-of-service traps.
- **Hypothesis:** The RL defender will achieve a detection rate exceeding 65% while maintaining service availability above 80%.

#### Scenario 4: `04_co_evolution.yaml` (Central Thesis Experiment)
- **Attacker:** `PPOAttacker` | **Defender:** `RLDefender`
- **Objective:** Investigates simultaneous dynamic co-evolution where both agents update policies concurrently in a 2-player Markov game.
- **Hypothesis:** The RL defender will preserve system resilience (availability ≥ 70%) even under continuous adversarial learning from the PPO attacker.

#### Scenario 5: `05_llm_vs_llm.yaml` (Exploratory Generative Frontier)
- **Attacker:** `LLMAttacker` (Google Gemini 3.5 Flash) | **Defender:** `LLMDefender` (Google Gemini 3.5 Flash)
- **Objective:** Explores autonomous generative AI reasoning in cyber conflict, pitting a red-team LLM agent planning multi-stage kill-chains against a blue-team LLM SOC analyst with physical ESP8266 IoT hardware in the loop.
- **Hypothesis:** The Gemini LLM defender will identify and mitigate Gemini-generated cyber kill-chain tactics with a detection rate of at least 60% while maintaining mean service availability above 0.75.


---

## 7. Target Environment & Physical Honeypot

The target environment models an interconnected corporate and IoT topology:

```
                               Target Network
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
      DMZ Subnet              Corporate Subnet            IoT Subnet
  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  │ - Public HTTP   │       │ - Internal SSH  │       │ - Simulated IoT │
  │ - Port 80, 8080 │       │ - Port 22       │       │ - NodeMCU ESP8266│
  └─────────────────┘       └─────────────────┘       └─────────────────┘
```

### Physical Hardware Honeypot (NodeMCU ESP8266)
In addition to simulated services, the framework integrates a physical **NodeMCU ESP8266 (ESP-12E)** microcontroller:
- **Firmware (`esp8266/esp8266.ino`):** Exposes multi-protocol web traps (`/admin`, `/login`, `/cgi-bin/status`, `/status`).
- **Telemetry Bridge (`securemesh_sce/environment/iot/hardware_bridge.py`):** Captures physical device telemetry (free heap memory, WiFi RSSI signal strength, uptime, connection states) and synchronizes with the game state.

#### Hardware Configuration:
1. Copy template:
   ```bash
   cp esp8266/secrets.h.example esp8266/secrets.h
   ```
2. Edit `esp8266/secrets.h` with your local Wi-Fi credentials and computer IP:
   ```cpp
   #define WIFI_SSID "YourLocalWiFi"
   #define WIFI_PASSWORD "YourWiFiPassword"
   #define BACKEND_URL "http://192.168.0.106:8000/api/logs/"
   ```
3. Compile and flash via PlatformIO:
   ```bash
   pio run -d esp8266 -t upload
   ```

---

## 8. Repository Architecture

```
honeypot/
├── backend/                      # FastAPI REST & WebSocket Backend
│   ├── app.py                    # Server entrypoint and route mounting
│   └── routes/
│       ├── experiment.py         # Live experiment execution & WebSocket streaming
│       └── logs.py               # Hardware honeypot log receiver
├── esp8266/                      # Physical IoT Honeypot Firmware
│   ├── esp8266.ino               # Arduino multi-trap firmware
│   ├── platformio.ini            # PlatformIO build configuration
│   └── secrets.h.example         # Template for private credentials
├── experiments/                  # Experiment Suite & Results
│   ├── matrix_config.yaml        # 2×2 experiment matrix specification
│   ├── results/                  # Generated telemetry, logs, and summaries
│   └── scenarios/                # Declarative scenario YAML definitions
│       ├── 01_baseline.yaml      # Baseline: Scripted vs Static
│       ├── 02_rl_attacker.yaml   # Offensive AI: PPO vs Static
│       ├── 03_rl_defender.yaml   # Defensive AI: Scripted vs RL
│       ├── 04_co_evolution.yaml  # Co-Evolution: PPO vs RL
│       └── 05_llm_vs_llm.yaml    # Generative Duel: LLM vs LLM (Gemini 3.5 Flash)
├── scripts/                      # Automation & Diagnostics
│   ├── run_hardware_suite.py     # Automated sequential runner for all 5 scenarios
│   └── diagnose_rl.py            # Diagnostic script for RL convergence & policies
├── securemesh_sce/               # Core Research Package
│   ├── agents/                   # Agent implementations
│   │   ├── attacker/             # ScriptedAttacker, PPOAttacker, LLMAttacker
│   │   ├── defender/             # StaticDefender, RLDefender, LLMDefender
│   │   └── common/               # PPO core, MLP networks, replay buffer, LLMClient
│   ├── environment/              # Simulated network & physical bridges
│   │   ├── iot/                  # ESP8266 simulator and physical bridge
│   │   ├── network/              # Topology, hosts, and services
│   │   ├── services/             # SSH and HTTP service models
│   │   └── telemetry/            # Metric collectors and anomaly detectors
│   ├── evaluation/               # Metrics and statistical analysis
│   │   ├── metrics.py            # EpisodeMetrics & MetricsEngine
│   │   ├── plots.py              # Publication-grade plotting utilities
│   │   └── statistics.py         # Hypothesis testing & confidence intervals
│   ├── experiments/              # Experiment lifecycle engine & runner
│   │   ├── engine.py             # SCENE multi-episode orchestration engine
│   │   └── runner.py             # CLI runner entrypoint
│   ├── game/                     # Game-Theoretic Markov Game Core
│   │   ├── actions.py            # Action definitions, impact tiers, attacker types
│   │   ├── bayesian_game.py      # Gymnasium-compatible Markov Game environment
│   │   ├── rewards.py            # Multi-objective reward functions
│   │   ├── state.py              # State vector representation
│   │   └── transitions.py        # State transition engine
│   └── logging/                  # Experiment and telemetry loggers
├── tests/                        # Automated Pytest Suite
│   ├── test_agents.py            # Tests for attacker and defender policies
│   ├── test_game.py              # Tests for Markov game transitions and rewards
│   └── test_scene_engine.py      # Tests for experiment engine and hypotheses
├── requirements.txt              # Unified Python dependencies
└── README.md                     # Comprehensive documentation
```

---

## 9. Installation & Quick Start

### 1. Prerequisites
- Python 3.10+
- PlatformIO (optional, required only for flashing physical ESP8266 hardware)
- Google Gemini API Key (optional, for Scenario 5) or local Ollama (`llama3.1:8b`)

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/itssaideep/SecureMesh.git
cd SecureMesh

# Create and activate Python virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables (create .env)
# ESP8266_DEVICE_IP=192.168.0.111
# GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run Unit Tests
```bash
python -m pytest tests/ -v
```
*Expected: 10 passed.*

### 4. Run Experiment Scenarios
Run any individual scenario using the CLI runner:

```bash
# Scenario 1: Baseline (Scripted vs Static)
python -m securemesh_sce.experiments.runner --config experiments/scenarios/01_baseline.yaml --episodes 10

# Scenario 2: RL Attacker (PPO vs Static)
python -m securemesh_sce.experiments.runner --config experiments/scenarios/02_rl_attacker.yaml --episodes 10

# Scenario 3: RL Defender (Scripted vs RL)
python -m securemesh_sce.experiments.runner --config experiments/scenarios/03_rl_defender.yaml --episodes 10

# Scenario 4: Adversarial Co-Evolution (PPO vs RL)
python -m securemesh_sce.experiments.runner --config experiments/scenarios/04_co_evolution.yaml --episodes 10

# Scenario 5: Generative Duel (LLM vs LLM via Gemini 3.5 Flash)
python -m securemesh_sce.experiments.runner --config experiments/scenarios/05_llm_vs_llm.yaml --episodes 5
```

### 5. Automated Physical Hardware Suite (All 5 Scenarios)
To run all 5 scenarios sequentially on the physical ESP8266 testbed and generate an aggregated comparative report:

```bash
python scripts/run_hardware_suite.py --episodes 5
```

Results are saved to `experiments/results/`:
- `*.telemetry.jsonl`: Step-by-step game telemetry records.
- `*.episodes.csv`: Per-episode performance metrics.
- `*.summary.json`: Aggregated metrics and statistical hypothesis verdict.
- `physical_hardware_suite_report.md`: Comparative Markdown report across all 5 scenarios.

---

## 10. REST & WebSocket API Reference

The testbed includes a FastAPI service supporting real-time experiment streaming:

### Start the Server
```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
- **Interactive Documentation:** `http://127.0.0.1:8000/docs`

### Key Endpoints
| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Service health status |
| `POST` | `/api/experiment/start` | Launch a background experiment run |
| `WS` | `/api/experiment/ws` | Real-time WebSocket stream of game steps and episode summaries |
| `POST` | `/api/logs/` | Ingest security events from physical ESP8266 honeypot |
| `GET` | `/api/logs/` | Query recent honeypot logs |

---

## 11. License & Citation

This project is open-source software licensed under the [MIT License](https://opensource.org/licenses/MIT).
