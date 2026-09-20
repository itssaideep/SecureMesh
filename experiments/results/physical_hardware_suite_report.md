# SecureMesh-SCE: Physical Hardware Experiment Suite (Scenarios 1-4)
**Generated at:** 2026-09-19 22:04:21  
**Hardware Device:** ESP8266 NodeMCU (`192.168.0.111`)  
**Backend Ingestion:** `http://192.168.0.106:8000/api/logs/`  

---

## Summary Results Table

| Scenario | Name | Attacker | Defender | ASR | Detection Rate | Availability | Cost | Hypothesis | Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `SCE-001` | Deterministic Baseline | SCRIPTED | STATIC | 0.1867 | 0.8300 | 1.0000 | 870.00 | `detection_rate above 0.5` | **PASS** |
| `SCE-002` | RL Attacker vs Static Defender | PPO | STATIC | 0.2249 | 0.9095 | 1.0000 | 837.00 | `attack_success_rate above 0.4` | **FAIL** |
| `SCE-003` | Scripted Attacker vs RL Defender | SCRIPTED | RL | 0.0233 | 0.1367 | 0.1254 | 702.00 | `detection_rate above 0.65` | **FAIL** |
| `SCE-004` | Adversarial Co-Evolution (RL vs RL) | PPO | RL | 0.0312 | 0.1086 | 0.1254 | 697.80 | `mean_service_availability above 0.7` | **FAIL** |

---

## Scenario Key Insights

1. **SCE-001 (Baseline: Scripted vs Static)**: Establishes the non-adaptive benchmark where fixed rule-based defenders face predictable kill-chains.
2. **SCE-002 (RL Attacker: PPO vs Static)**: Demonstrates the ability of an adaptive PPO policy to discover evasion paths against rigid static rules.
3. **SCE-003 (RL Defender: Scripted vs RL)**: Evaluates whether an autonomous RL agent can suppress attack impact while avoiding unnecessary service downtime.
4. **SCE-004 (Co-Evolution: PPO vs RL)**: Simulates simultaneous two-player Markov game dynamics with both offensive and defensive policies adapting concurrently.

---

All telemetry, episode summaries, and hypothesis validations were recorded with live physical hardware in the loop.