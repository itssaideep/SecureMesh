# SecureMesh-SCE

**A SCENE-Guided, AI-Assisted Security Chaos Engineering Testbed for Studying Adaptive Cyberattack and Cyberdefence in IoT Environments.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React: Vite](https://img.shields.io/badge/Frontend-React%20%7C%20Vite%20%7C%20Tailwind-61DAFB.svg)](frontend/)
[![Hardware: ESP8266](https://img.shields.io/badge/Hardware-NodeMCU%20ESP8266-red.svg)](esp8266/)
[![LLM: Ollama & Gemini](https://img.shields.io/badge/Offensive%20LLM-Llama%203.1%20%7C%20Gemini%202.0-8A2BE2.svg)](securemesh_sce/agents/attacker/llm_policy.py)
[![Tests: 12 Passed](https://img.shields.io/badge/Tests-12%20Passed-brightgreen.svg)](tests/)

---

## Table of Contents
1. [Research Overview & Theoretical Foundation](#1-research-overview--theoretical-foundation)
2. [Complete System Architecture](#2-complete-system-architecture)
3. [Repository Directory Structure](#3-repository-directory-structure)
4. [Agent Hierarchies & Action Spaces](#4-agent-hierarchies--action-spaces)
5. [Bayesian Formulation & Belief Tracking](#5-bayesian-formulation--belief-tracking)
6. [Physical ESP8266 Hardware Honeypot](#6-physical-esp8266-hardware-honeypot)
7. [Offensive LLM Integration (Ollama & Gemini)](#7-offensive-llm-integration-ollama--gemini)
8. [The 8-Stage SCENE Experiment Lifecycle](#8-the-8-stage-scene-experiment-lifecycle)
9. [Canonical Scenarios & Experimental Protocol](#9-canonical-scenarios--experimental-protocol)
10. [Empirical Evaluation Matrix & Key Findings](#10-empirical-evaluation-matrix--key-findings)
11. [Quick Start & Installation](#11-quick-start--installation)
12. [Full-Stack Execution (Backend, Frontend, Docker)](#12-full-stack-execution-backend-frontend-docker)
13. [Artifact Generation & Verification](#13-artifact-generation--verification)
14. [TAFFAC Threat Modeling & Risk Register](#14-taffac-threat-modeling--risk-register)
15. [Research Documentation Index](#15-research-documentation-index)
16. [License & Citation](#16-license--citation)

---

## 1. Research Overview & Theoretical Foundation

**SecureMesh-SCE** is an open-source, reproducible research platform engineered strictly following the **SCENE guidelines** (*Security Chaos Engineering in Networked Environments*, Jolak et al., 2026) and the **TAFFAC threat assessment framework** (*Threat Assessment Framework for Fog and Cloud*, Sood et al., 2026). It investigates the foundational research question:

> **Core Research Question:** *How does epistemic uncertainty regarding adversary strategic intent and technical capability impact the resilience, safety, and effectiveness of adaptive AI-based cyberdefence in heterogeneous IoT networks?*

### Problem Statement
In real-world Internet of Things (IoT) deployments, defenders operate under deep asymmetric uncertainty. The adversary's technical sophistication, stealth budget, persistence, and underlying attack graphs are unobservable. 
- **Static heuristic defenses** (e.g., rigid Snort/Suricata rules, fixed fail2ban thresholds) fail rapidly against polymorphic, adaptive, or imitation-based kill-chains.
- **Naive Reinforcement Learning (RL) defenders** trained without explicit belief modeling frequently overreact to noisy probes by aggressively severing communication links or shutting down services, inducing catastrophic self-inflicted Denial-of-Service (DoS) outages.
- **SecureMesh-SCE** resolves this trade-off by formalizing the adversary-defender interaction as a **two-player Bayesian Markov Game**, pitting an offensive capability spectrum (deterministic scripts, imitation learning, autonomous RL, and generative LLMs) against a defensive hierarchy that maintains an exact Bayesian posterior distribution over latent attacker profiles, filtered through a deterministic **Safety Gate**.

---

## 2. Complete System Architecture

```
                                  SecureMesh-SCE Research Testbed
                                                 |
                       +-------------------------+-------------------------+
                       |                                                   |
              OFFENSE HIERARCHY                                   DEFENCE HIERARCHY
          (Levels 0–4 Policy Space)                           (Levels 1–5 Policy Space)
          - Level 0: Scripted (Deterministic)                 - Level 1: Static (Heuristic Rules)
          - Level 1: BC (Behavioral Cloning)                  - Level 2: ML (Random Forest)
          - Level 2: GAIL (Imitation Learning)                - Level 3: RL (PPO on System State)
          - Level 3: Adaptive PPO                             - Level 4: Bayesian RL (PPO + Belief)
          - Level 4: Generative LLM (Llama / Gemini)          - Level 5: Constrained (Safety Gate)
                       |                                                   |
                       +-------------------------+-------------------------+
                                                 |
                                     BAYESIAN MARKOV GAME
                             State: s_t = [x_t, b_t, h_t, r_t] in R^32
                                                 |
                             +-------------------+-------------------+
                             |                                       |
                     SIMULATED TESTBED                       PHYSICAL MESH
                   - Subnet Topology (DMZ, Corp, IoT)      - NodeMCU ESP8266 (Port 80)
                   - SSH Honeypot (Cowrie Semantics)       - ESP32 Gateway Node
                   - HTTP/CGI Web Service                  - Local Mosquitto MQTT Broker
                   - Synthetic IoT Sensors                 - Docker Containerized Services
                             |                                       |
                             +-------------------+-------------------+
                                                 |
                                     SCENE EXPERIMENT ENGINE
                              (8-Stage Chaos Experiment Lifecycle)
                                                 |
                       +-------------------------+-------------------------+
                       |                                                   |
               EVALUATION ENGINE                                  DYNAMIC RISK REGISTER
          - Security (ASR, Detection Rate, F1)               - TAFFAC Attribute Taxonomy
          - Response (MTTD, MTTR, Defense Cost)              - STRIDE Threat Categorization
          - Bayesian (Brier Score, ECE, Log Loss)            - MITRE ATT&CK Technique Mapping
          - Resilience (Availability, Impact Tiers)          - Dynamic Residual Risk Calculation
          - Hypothesis Testing (Mann-Whitney U, Cohen's d)
```

---

## 3. Repository Directory Structure

```
SecureMesh/
├── backend/                      # FastAPI REST & WebSocket Backend
│   ├── app.py                    # Application entrypoint & CORS configuration
│   ├── routes/
│   │   ├── experiment.py         # Live SCENE experiment triggers & WebSocket stream
│   │   ├── ai_analysis.py        # Threat analysis endpoint
│   │   └── logs.py               # Structured log querying
│   └── requirements.txt          # Backend server dependencies
│
├── frontend/                     # React + Vite Dashboard
│   ├── src/
│   │   ├── App.jsx               # Real-time threat & telemetry dashboard
│   │   └── main.jsx              # UI mount point
│   ├── package.json              # NPM dependencies (Vite, Tailwind, Chart.js)
│   └── tailwind.config.js        # Dashboard styling tokens
│
├── esp8266/                      # Physical Hardware Honeypot Firmware
│   ├── esp8266.ino               # Arduino C++ honeypot traps (HTTP/CGI/Login/Status)
│   └── platformio.ini            # PlatformIO build & flashing configuration
│
├── securemesh_sce/               # Core Research Framework (Python Package)
│   ├── agents/
│   │   ├── attacker/             # Offense Spectrum: Scripted, BC, GAIL, PPO, LLM
│   │   └── defender/             # Defense Spectrum: Static, ML, RL, Bayesian RL, Constrained
│   ├── environment/
│   │   ├── iot/                  # Simulated ESP8266/ESP32 & PhysicalESP8266Bridge
│   │   ├── network/              # Multi-subnet topology (DMZ, Corp, IoT)
│   │   ├── services/             # SSH, HTTP, and MQTT emulated services
│   │   └── telemetry/            # Chaos telemetry collector & metrics aggregator
│   ├── evaluation/               # Metrics, statistics, LaTeX tables & matplotlib plots
│   ├── experiments/              # SCENE 8-stage experiment engine & matrix runners
│   ├── game/                     # Gymnasium Bayesian Markov Game (States, Actions, Rewards)
│   ├── inference/                # Bayesian posterior tracker & calibration evaluators
│   └── risk/                     # TAFFAC threat model & dynamic risk register
│
├── experiments/
│   ├── scenarios/                # Canonical scenario configs (01_baseline to 08_llm_offensive)
│   └── results/                  # Generated paper artifacts (LaTeX, SVG/PNG, CSV, JSONL)
│
├── scripts/
│   ├── run_all_ablations.py      # Automated multi-seed execution of 5x5 matrix
│   └── generate_paper_artifacts.py # Compiles LaTeX tables, figures, and risk registers
│
├── docs/                         # Peer-Review Documentation Suite
│   ├── architecture.md           # System components & data flow
│   ├── methodology.md            # Bayesian Markov Game formalization
│   ├── threat_model.md           # TAFFAC framework & STRIDE mapping
│   ├── experiments.md            # 8-stage SCENE scenarios & hypothesis catalog
│   ├── reproducibility.md        # Step-by-step artifact reproduction guide
│   └── limitations.md            # Threats to validity & hardware boundary conditions
│
├── tests/                        # Comprehensive unit & integration tests (12 passing)
├── docker-compose.yml            # Containerized Cowrie SSH & MongoDB instances
├── requirements.txt              # Unified research dependencies
└── README.md                     # This file
```

---

## 4. Agent Hierarchies & Action Spaces

### Offense Hierarchy (Levels 0–4)
| Level | Identifier | Implementation | Strategic Characteristics |
|:---:|:---|:---|:---|
| **L0** | `ScriptedAttacker` | Deterministic Kill-chain | Executes fixed reconnaissance $\to$ brute-force $\to$ exploit $\to$ persist progression. |
| **L1** | `BCAttacker` | Behavioral Cloning | Supervised MLP policy trained on historical human/scripted compromise trajectories. |
| **L2** | `GAILAttacker` | Generative Adversarial Imitation | Adversarial discriminator minimizing Jensen-Shannon divergence to expert state-action occupancy. |
| **L3** | `PPOAttacker` | Adaptive Deep RL | PPO agent trained against static defense to exploit blind spots and evade detection. |
| **L4** | `LLMAttacker` | Generative LLM Planner | Zero-shot / few-shot dynamic reasoning via **Ollama (Llama 3.1:8b)** or **Google Gemini 2.0 Flash**. |

#### Attacker Action Space ($|\mathcal{A}_A| = 13$)
`Recon`, `PortScan`, `VulnerabilityScan`, `BruteForceSSH`, `WebExploit`, `IoTBufferOverflow`, `LateralMovement`, `PrivilegeEscalation`, `C2Beacon`, `ExfiltrateData`, `ClearLogs`, `EvasionTechnique`, `DoSFlood`.

---

### Defence Hierarchy (Levels 1–5)
| Level | Identifier | Implementation | Strategic Characteristics |
|:---:|:---|:---|:---|
| **D1** | `StaticDefender` | Heuristic Rules | Threshold-triggered responses (e.g., fail2ban after 3 failed logins). |
| **D2** | `RandomForestDefender` | Supervised ML | Multi-class Random Forest classifier mapping telemetry windows to mitigation actions. |
| **D3** | `RLDefender` | Un-augmented PPO | Deep RL agent observing only physical network state $x_t$ without belief tracking. |
| **D4** | `BayesianRLDefender` | Bayesian PPO | PPO policy conditioned on the full augmented state $s_t = [x_t, b_t, h_t, r_t]$. |
| **D5** | `ConstrainedDefender` | Safety-Gated Bayesian RL | Bayesian RL policy filtered through an action Safety Gate that prevents self-inflicted outages. |

#### Defender Action Space ($|\mathcal{A}_D| = 14$)
`NoAction`, `Monitor`, `RateLimit`, `HoneypotRedirect`, `BlockIP`, `TerminateSession`, `PatchVulnerability`, `RotateCredentials`, `IsolateIoTDevice`, `SegmentNetwork`, `RestartService`, `RollbackState`, `DeceiveAttacker`, `IncreaseLogging`.

#### Action Impact Tiers & Safety Gate
- **Low Impact (Tier 1)**: `NoAction`, `Monitor`, `IncreaseLogging`. Always permitted.
- **Medium Impact (Tier 2)**: `RateLimit`, `HoneypotRedirect`, `DeceiveAttacker`, `RotateCredentials`. Permitted when risk score $> 0.25$.
- **High Impact (Tier 3)**: `BlockIP`, `TerminateSession`, `PatchVulnerability`, `RestartService`. Requires confirmed adversary belief ($\max b_t > 0.40$).
- **Critical Impact (Tier 4)**: `IsolateIoTDevice`, `SegmentNetwork`, `RollbackState`. Strictly blocked unless attack is confirmed and verified ($\max b_t > 0.65$ and risk $> 0.70$).

---

## 5. Bayesian Formulation & Belief Tracking

The interaction is modeled as a partially observable Bayesian Markov Game:
$$\mathcal{M} = \langle \mathcal{S}, \Theta, \mathcal{A}_A, \mathcal{A}_D, \mathcal{T}, \mathcal{R}_A, \mathcal{R}_D, \mathcal{O}, \Omega, \gamma \rangle$$

### State Vector Representation ($s_t \in \mathbb{R}^{32}$)
At each discrete timestep $t$, the defender observes:
$$s_t = [x_t \in \mathbb{R}^{16} \parallel b_t \in \mathbb{R}^{4} \parallel h_t \in \mathbb{R}^{8} \parallel r_t \in \mathbb{R}^{4}]$$
1. **$x_t$ (System Telemetry)**: CPU load, memory utilization, packet drop rate, active sessions, failed auth count, open ports, alert status.
2. **$b_t$ (Bayesian Belief Vector)**: Posterior probability over the latent adversary type space:
   $$\Theta = \{\theta_1: \text{opportunistic}, \, \theta_2: \text{stealth}, \, \theta_3: \text{adaptive}, \, \theta_4: \text{resource\_aware}\}$$
3. **$h_t$ (History Embedding)**: Exponential moving averages of recent attacker action types and impact levels.
4. **$r_t$ (Risk Context)**: Current operational asset values, service availability, and aggregate threat impact.

### Exact Posterior Update
$$\mathbb{P}(\theta_k \mid o_{1:t}) \propto \mathbb{P}(o_t \mid \theta_k, s_t) \cdot \mathbb{P}(\theta_k \mid o_{1:t-1})$$

The belief tracker calibration is evaluated using:
- **Brier Score**: $\text{BS} = \frac{1}{T} \sum_{t=1}^T \sum_{k=1}^K (b_t(k) - y_t(k))^2$
- **Expected Calibration Error (ECE)**: Binning predictions into $M=10$ intervals to quantify reliability.

---

## 6. Physical ESP8266 Hardware Honeypot

The platform seamlessly bridges simulated software nodes with **physical NodeMCU ESP8266 hardware**.

### Hardware Requirements
- **Microcontroller**: NodeMCU v2 / v3 (ESP8266-12E) with Micro-USB data cable.
- **USB-UART Driver**: Silicon Labs CP210x or WCH CH340 driver installed on the host.
- **Wi-Fi Network**: **2.4 GHz 802.11 b/g/n only**. (ESP8266 hardware does not support 5 GHz or 6 GHz Wi-Fi).

### Firmware Trap Architecture (`esp8266/esp8266.ino`)
The ESP8266 runs an asynchronous web server exposing emulated IoT endpoints:
- `GET /`: Deceptive smart-plug admin dashboard.
- `POST /login`: Credential harvesting trap (captures brute-force user/pass, logs alert, returns HTTP 401).
- `GET /cgi-bin/status?cmd=...`: Command injection trap detecting remote code execution probes.
- `GET /status`: Live telemetry stream providing free heap bytes, Wi-Fi RSSI (dBm), and uptime (seconds).

### Flashing via PlatformIO
1. Configure Wi-Fi credentials in `esp8266/esp8266.ino`:
   ```cpp
   const char* ssid     = "Your_2.4GHz_SSID";
   const char* password = "Your_Password";
   ```
2. Build and flash the firmware:
   ```powershell
   # Flash over USB serial (COM6 or auto-detected port)
   pio run -d esp8266 --target upload

   # Launch serial monitor at 115200 baud
   pio device monitor -b 115200 --port COM6
   ```
3. Read the assigned IP address from serial output (e.g., `10.174.55.242`).

### Hardware Bridge Integration
Connect your live physical board directly into the Bayesian Markov Game:
```python
from securemesh_sce.environment.iot import PhysicalESP8266Bridge

# Initialize bridge to physical device IP
bridge = PhysicalESP8266Bridge(device_ip="10.174.55.242")

print("Device Reachable:", bridge.is_reachable())
print("Live Telemetry:", bridge.get_hardware_telemetry())
# Returns: {'free_heap': 44128, 'rssi_dbm': -58, 'uptime_sec': 142, 'device_type': 'esp8266_nodemcu'}
```

---

## 7. Offensive LLM Integration (Ollama & Gemini)

Level 4 adversaries utilize generative language models to reason over defensive counter-measures and generate adaptive exploit sequences.

```
       Current State s_t  -----> [ Structured System Prompt ]
                                             |
                                  [ LLM Reasoning Engine ]
                                  (Ollama or Google Gemini)
                                             |
    Next Action a_t^A    <----- [ Strict JSON Schema Validation ]
                                 (Fallback to Heuristic on Error)
```

### Option A: Local LLM via Ollama (Llama 3.1:8b)
1. Install and run [Ollama](https://ollama.ai/):
   ```powershell
   ollama pull llama3.1:8b
   ollama run llama3.1:8b
   ```
2. Set your environment variable:
   ```powershell
   $env:LLM_PROVIDER="ollama"
   $env:OLLAMA_MODEL="llama3.1:8b"
   ```
3. Run the LLM scenario:
   ```powershell
   python -m securemesh_sce.experiments.runner --config experiments/scenarios/08_llm_offensive.yaml --episodes 5
   ```

### Option B: Cloud LLM via Google Gemini
1. Add your API key to `.env`:
   ```env
   GEMINI_API_KEY=your_google_ai_studio_api_key_here
   GEMINI_MODEL=gemini-2.0-flash
   LLM_PROVIDER=gemini
   ```
2. Run the scenario:
   ```powershell
   python -m securemesh_sce.experiments.runner --config experiments/scenarios/08_llm_offensive.yaml --episodes 5
   ```

---

## 8. The 8-Stage SCENE Experiment Lifecycle

All experiments execute deterministically through the 8-stage SCENE lifecycle:

```
[ Stage 1: Hypothesis ] ──> Pre-register threshold (e.g., ASR <= 0.20, Availability >= 0.95)
         │
[ Stage 2: Steady State ] ─> Verify baseline system invariants (Availability >= 0.99)
         │
[ Stage 3: Inject Chaos ] ─> Inject attacker actions, service faults, or network jitter
         │
[ Stage 4: Joint Action ] ─> Simultaneous transition: s_{t+1} ~ T(s_t, a_t^A, a_t^D)
         │
[ Stage 5: Telemetry ] ───> Stream packet drops, CPU load, and IDS alerts to collector
         │
[ Stage 6: Evaluate ] ────> Compute statistical verdict (PASS / FAIL) with Mann-Whitney U
         │
[ Stage 7: Risk Register] ─> Update TAFFAC control maturity and residual risk scores
         │
[ Stage 8: Export ] ──────> Emit JSONL telemetry, CSV episode summaries, and LaTeX tables
```

---

## 9. Canonical Scenarios & Experimental Protocol

| Scenario | Config File | Attacker | Defender | Primary Research Question |
|:---:|:---|:---:|:---:|:---|
| **Exp 1** | `01_baseline.yaml` | Scripted (L0) | Static (D1) | How do rule-based baseline defenses perform against deterministic kill-chains? |
| **Exp 2** | `02_defense_rl.yaml` | Scripted (L0) | RL (D3) | Can un-augmented deep RL learn effective mitigation policies from raw telemetry? |
| **Exp 3** | `03_imitation.yaml` | BC (L1) | Static (D1) | Do imitation-learning attackers successfully circumvent static rule thresholds? |
| **Exp 4** | `04_attacker_adaptation.yaml` | PPO (L3) | Static (D1) | How quickly does an adaptive RL adversary discover defensive blind spots? |
| **Exp 5** | `05_co_adaptation.yaml` | PPO (L3) | RL (D3) | What equilibrium emerges under simultaneous attacker-defender reinforcement learning? |
| **Exp 6** | `06_bayesian_defense.yaml` | PPO (L3) | Bayesian RL (D4) | Does explicit Bayesian belief tracking reduce mean time to detect and mitigate? |
| **Exp 7** | `07_constrained_safety.yaml` | PPO (L3) | Constrained (D5) | Does an action safety gate eliminate self-inflicted service availability outages? |
| **Exp 8** | `08_llm_offensive.yaml` | LLM (L4) | Bayesian RL (D4) | Can Bayesian RL mitigate zero-day kill-chains generated by LLM reasoning engines? |

---

## 10. Empirical Evaluation Matrix & Key Findings

The complete $5 \times 5$ cross-product evaluation matrix evaluates all Attacker Levels against all Defender Levels across multiple seeds:

### Attack Success Rate (ASR) $\downarrow$ vs. Service Availability $\uparrow$
| Attacker Level | Static (D1) | ML (D2) | RL (D3) | Bayesian RL (D4) | Constrained (D5) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Scripted (L0)** | 0.19 / 1.00 | 0.14 / 0.98 | 0.11 / 0.96 | 0.08 / 0.99 | **0.06 / 1.00** |
| **BC Imitation (L1)** | 0.38 / 0.91 | 0.31 / 0.92 | 0.26 / 0.90 | 0.18 / 0.96 | **0.12 / 0.99** |
| **GAIL (L2)** | 0.47 / 0.86 | 0.39 / 0.88 | 0.33 / 0.85 | 0.22 / 0.94 | **0.15 / 0.98** |
| **Adaptive PPO (L3)** | 0.62 / 0.72 | 0.51 / 0.81 | 0.44 / 0.78 | 0.27 / 0.89 | **0.18 / 0.96** |
| **LLM-Assisted (L4)** | 0.55 / 0.76 | 0.46 / 0.83 | 0.38 / 0.82 | 0.24 / 0.91 | **0.16 / 0.97** |

### Key Experimental Insights
1. **Uncertainty Penalty**: Against an adaptive PPO adversary (L3), static defenses fail with a $0.62$ ASR. While un-augmented RL (D3) reduces ASR to $0.44$, it degrades availability down to $0.78$ due to unconstrained IP blocking.
2. **Bayesian Belief Advantage**: Bayesian RL (D4) cuts ASR down to $0.27$ while maintaining $0.89$ availability, achieving faster convergence by conditioning responses on adversary profile classification.
3. **Safety Gate Pareto Dominance**: Constrained Defender (D5) achieves the lowest overall ASR ($0.18$ vs L3, $0.16$ vs L4) while preserving near-perfect availability ($0.96\text{--}0.97$), proving that bounded autonomous defense prevents self-inflicted outages.

---

## 11. Quick Start & Installation

### Prerequisites
- Python 3.10 or higher
- Git
- Node.js 18+ (for frontend dashboard)
- Docker & Docker Compose (optional, for Cowrie honeypot & MongoDB)

### Setup
```bash
# 1. Clone repository
git clone https://github.com/itssaideep/SecureMesh.git
cd SecureMesh

# 2. Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
# source venv/bin/activate

# 3. Install research dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### Run Unit Tests
Verify all components are operational:
```bash
python -m pytest tests/ -v
# Expected: 12 passed in < 5.0s
```

---

## 12. Full-Stack Execution (Backend, Frontend, Docker)

### 1. Launch Emulated Infrastructure (Optional)
```bash
docker-compose up -d
# Launches Cowrie SSH honeypot on ports 2222/2223 and MongoDB on 27017
```

### 2. Launch FastAPI Backend Server
```powershell
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/`
- WebSocket live experiment stream: `ws://localhost:8000/api/experiment/ws`

### 3. Launch React + Vite Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
- Open browser at `http://localhost:5173` to view the real-time telemetry dashboard.

### 4. Trigger Live Web Experiment via REST API
```bash
curl -X POST http://localhost:8000/api/experiment/start \
  -H "Content-Type: application/json" \
  -d "{\"attacker\": \"ppo\", \"defender\": \"bayesian_rl\", \"scenario\": \"adaptive_defense\", \"episodes\": 5}"
```

---

## 13. Artifact Generation & Verification

### Run Single Experiment via CLI
```powershell
# Run Bayesian Defense Scenario (Exp 6)
python -m securemesh_sce.experiments.runner --config experiments/scenarios/06_bayesian_defense.yaml --episodes 10

# Run Safety Gate Scenario (Exp 7)
python -m securemesh_sce.experiments.runner --config experiments/scenarios/07_constrained_safety.yaml --episodes 10
```

### Run Automated Multi-Seed Matrix & Ablation Suite
```powershell
python scripts/run_all_ablations.py --episodes 5 --seeds 42 101 --duration 50
```

### Generate Publication Figures & LaTeX Tables
```powershell
python scripts/generate_paper_artifacts.py
```
Generated artifacts in `experiments/results/`:
- `tab_adversary_matrix.tex`: Complete LaTeX booktabs cross-product table.
- `fig_belief_trajectory.png` / `.svg`: Posterior belief convergence trajectories over time.
- `fig_calibration_curve.png`: Reliability diagram and Expected Calibration Error.
- `fig_ablation_comparison.png`: Bar chart comparing defensive ablations across metrics.
- `risk_register.md`: Markdown export of updated dynamic risk scores.

---

## 14. TAFFAC Threat Modeling & Risk Register

SecureMesh-SCE integrates the **TAFFAC** threat assessment framework mapped directly to **STRIDE** and **MITRE ATT&CK**:

| Risk ID | Threat Description | STRIDE Category | MITRE ATT&CK | Initial Risk | Deployed Control | Residual Risk |
|:---|:---|:---|:---|:---:|:---|:---:|
| **RSK-01** | SSH credential brute-force stuffing | Spoofing | T1110 | 16 (L4×I4) | Rate Limiting & Cowrie Redirection | **8.00** |
| **RSK-02** | Web gateway command injection | Elevation of Privilege | T1059 | 20 (L4×I5) | Service Isolation & Input Sanitization | **11.67** |
| **RSK-03** | IoT endpoint buffer overflow & OOM crash | Denial of Service | T1499 | 15 (L5×I3) | Firmware Heap Monitoring & Isolation | **10.00** |
| **RSK-04** | Lateral movement across mesh subnets | Elevation of Privilege | T1021 | 12 (L3×I4) | Micro-segmentation & Dynamic IP Quarantine | **6.00** |
| **RSK-05** | Stealthy C2 beaconing & data exfiltration | Information Disclosure | T1041 | 15 (L3×I5) | Bayesian Anomaly Tracking & IDS Tuning | **8.75** |

Residual risk is dynamically updated following each SCENE experiment cycle based on empirical detection rate and control maturity:
$$\text{Residual Risk} = \text{Initial Risk} \times \left(1 - \frac{\text{Maturity}}{5.0} \times \text{Detection Rate}\right)$$

---

```
