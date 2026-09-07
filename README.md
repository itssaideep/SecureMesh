# BASTION

**Bayesian Adversarial Simulation Testbed for IoT Networks**

BASTION is a research-oriented cybersecurity testbed that formulates the interaction between an attacker and a defender as a Bayesian Markov game. It enables researchers to simulate network topologies, vulnerable IoT services (ESP32/ESP8266), and intrusion detection systems (IDS) in a reproducible, software-only environment.

> **Research Question:** How can a defender make effective adaptive security decisions when the capabilities and behavior of an AI-based attacker are uncertain?

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

| Category | Details |
|---|---|
| **AI vs. AI Simulations** | Pit different agent types against each other — `RandomAttacker`, `ScriptedAttacker` (kill-chain), `RLAttacker` (adaptive PPO) vs. `StaticDefender` (heuristics), `MLDefender` (Random Forest), `RLDefender` (adaptive PPO). |
| **Bayesian Belief Tracking** | The RL Defender maintains and dynamically updates a probability distribution over attacker type (opportunistic, sophisticated, stealthy) based on observed behavior. |
| **Realistic IoT Simulation** | Simulated ESP8266 and ESP32 endpoints mimicking real-world vulnerabilities (MQTT, CoAP, OTA, credential-capture) — no physical hardware required. |
| **Evaluation Framework** | Built-in metrics (Attack Success Rate, Detection Rate, System Resilience, Adaptation KL Divergence), structured logging, and automated Matplotlib visualizations. |
| **React Dashboard** | Frontend for visualizing experiment results, reward curves, and belief evolution *(work in progress)*. |

---

## Architecture

```
securemesh_testbed/          Core Python package
├── agents/                  Attacker & defender implementations (incl. NumPy PPO)
├── environment/             Network simulators, IDS engines, IoT firmware mocks
├── game/                    Gymnasium-compatible Markov game formulation
├── experiments/             CLI runners and scenario definitions
└── evaluation/              Metrics, structured logging, plotting utilities

frontend/                    React + Vite dashboard (TailwindCSS)
backend/                     FastAPI server (legacy, preserved for reference)
esp8266/                     C++ firmware (legacy, preserved for reference)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js (for the frontend dashboard)

### Installation

A virtual environment is recommended. The testbed relies on lightweight dependencies — no heavy PyTorch requirement.

```bash
cd securemesh_testbed
pip install numpy gymnasium matplotlib pandas scikit-learn tabulate
```

---

## Usage

Run a simulated experiment from the CLI. The following example pits a scripted kill-chain attacker against an adaptive RL defender:

```bash
python -m securemesh_testbed.experiments.run_experiment \
    --scenario known_attacks \
    --attacker scripted \
    --defender rl \
    --episodes 50 \
    --seed 42 \
    --output securemesh_testbed/results/my_first_experiment
```

The experiment runner will execute the Markov game, train the RL agents if applicable, and generate CSV logs and PNG plots in the specified output directory.

---

## Results

Navigate to the output directory to view the generated plots:

| File | Description |
|---|---|
| `reward_curves.png` | Compares attacker and defender cumulative rewards over time. |
| `belief_evolution.png` | Shows how the defender's Bayesian belief about the attacker type evolves during an episode. |
| `detection_rate.png` | Detection rate of the IDS over time. |
| `service_availability.png` | Service availability across simulated endpoints. |

> **Note:** An interactive React dashboard for exploring these results is under active development.

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).
