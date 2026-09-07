# SecureMesh-SCE Architecture

## Overview

SecureMesh-SCE is an AI-assisted Security Chaos Engineering (SCE) testbed for studying adaptive attack and defence in IoT environments. It extends the SecureMesh honeypot into a controlled, reproducible security-chaos environment following the SCENE methodology (Jolak et al.).

## System Architecture

```
                     SecureMesh-SCE
                          │
            ┌─────────────┴─────────────┐
            │                           │
      AI ATTACKER                  AI DEFENDER
      (Random/Scripted/            (Static/ML/RL)
       Aggressive/Stealthy/              │
       ReconHeavy/RL/LLM)         Bayesian belief
            │                     tracking
       chooses action                   │
            │                    chooses response
            └─────────────┬─────────────┘
                          ▼
                ┌──────────────────┐
                │   IoT TESTBED    │
                │                  │
                │ SSH Service      │
                │ HTTP Service     │
                │ ESP32 Sim (MQTT) │
                │ ESP8266 Sim      │
                │ IDS Engine       │
                └────────┬─────────┘
                         │
                    telemetry
                         ▼
               Experiment Controller
               (SCE Engine)
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         Metrics                Experiment Record
         (ASR, DR, MTTD,       (JSON + CSV + JSONL)
          MTTR, Resilience)           │
              └──────────┬──────────┘
                         ▼
                 Matrix Runner
                 (Statistical analysis,
                  LaTeX/Markdown tables)
```

## Component Map

| Component | Location | Description |
|---|---|---|
| Markov Game Environment | `securemesh_testbed/game/` | Gymnasium-compatible two-player game |
| Attacker Agents | `securemesh_testbed/agents/attacker/` | 7 attacker types |
| Defender Agents | `securemesh_testbed/agents/defender/` | 3 defender types |
| Network Simulation | `securemesh_testbed/environment/` | Hosts, services, IoT devices, IDS |
| SCE Engine | `securemesh_testbed/experiments/engine.py` | YAML-driven experiment lifecycle |
| Matrix Runner | `securemesh_testbed/experiments/matrix_runner.py` | Attacker × Defender grid runner |
| Metrics & Evaluation | `securemesh_testbed/evaluation/` | Metrics, logging, plotting, comparison |
| Scenarios | `experiments/scenarios/` | YAML experiment definitions |

## Agent Types

### Attackers

| Agent | Type | Description |
|---|---|---|
| `RandomAttacker` | Baseline | Uniform random action selection |
| `ScriptedAttacker` | Deterministic | Kill-chain phase cycling |
| `AggressiveAttacker` | Deterministic | Fast escalation, 2-step phases |
| `StealthyAttacker` | Deterministic | Slow with NOOP/evasion padding |
| `ReconHeavyAttacker` | Deterministic | 70%+ steps on reconnaissance |
| `RLAttacker` | Adaptive | NumPy PPO with online learning |
| `LLMAttacker` | AI-Assisted | Gemini/Ollama with safety filter |

### Defenders

| Agent | Type | Description |
|---|---|---|
| `StaticDefender` | Rule-based | Fixed heuristic (block after N alerts) |
| `MLDefender` | Supervised | Random Forest trained on oracle data |
| `RLDefender` | Adaptive | PPO + Bayesian attacker-type belief |

## Research Questions

| RQ | Question |
|---|---|
| RQ1 | Does an RL attacker achieve higher attack success than a deterministic attacker? |
| RQ2 | Does adaptive defence reduce attack success and recovery time? |
| RQ3 | Does repeated attacker–defender interaction improve system resilience? |

## Experiment Matrix

```
              Defender
            Static    ML    RL
Attacker  ┌─────────────────────
Scripted  │   A       B     C
Aggressive│   D       E     F
Stealthy  │   G       H     I
ReconHeavy│   J       K     L
RL        │   M       N     O
LLM       │   P       Q     R
```

Each cell runs N seeds × M episodes with statistical comparison.
