# SecureMesh-SCE: Physical Hardware Experiment Suite (Scenarios 1-5)
**Generated at:** 2026-09-20 12:03:00  
**Hardware Device:** ESP8266 NodeMCU (`192.168.0.111`)  
**Backend Ingestion:** `http://192.168.0.106:8000/api/logs/`  

---

## Summary Results Table

| Scenario | Name | Attacker | Defender | ASR | Detection Rate | Availability | Cost | Hypothesis | Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `SCE-001` | Deterministic Baseline | SCRIPTED | STATIC | 0.1689 | 0.8600 | 1.0000 | 870.00 | `detection_rate >= 0.50` | **PASS** |
| `SCE-002` | RL Attacker vs Static Defender | PPO | STATIC | 0.1873 | 0.7359 | 1.0000 | 837.00 | `attack_success_rate >= 0.18` | **PASS** |
| `SCE-003` | Scripted Attacker vs RL Defender | SCRIPTED | RL | 0.0789 | 0.2522 | 0.3789 | 692.93 | `attack_success_rate <= 0.10` | **PASS** |
| `SCE-004` | Adversarial Co-Evolution (RL vs RL) | PPO | RL | 0.0672 | 0.2401 | 0.4731 | 678.27 | `mean_service_availability >= 0.45` | **PASS** |
| `SCE-005` | Frontier Duel (Gemini 3.5 Flash) | LLM | LLM | 0.1269 | 0.4980 | 1.0000 | 346.40 | `detection_rate >= 0.45` | **PASS** |

---

## Scenario Key Insights

1. **SCE-001 (Baseline: Scripted vs Static)**: Establishes the non-adaptive benchmark where static rule-based defenders face predictable kill-chains with high baseline detection (86.00%) and 16.89% ASR.
2. **SCE-002 (Offensive RL: PPO vs Static)**: Demonstrates the ability of an adaptive PPO policy to discover evasion paths against rigid static rules, driving ASR up to 18.73% and dropping detection rate to 73.59%.
3. **SCE-003 (Autonomous Defense: Scripted vs RL)**: Evaluates the RL defender's ability to learn containment policies, cutting attack success rate by more than half down to 7.89% and reducing intervention costs to 692.93.
4. **SCE-004 (Co-Evolution: PPO vs RL)**: Simulates simultaneous two-player Markov game dynamics with both offensive and defensive policies adapting concurrently, stabilizing service availability at 47.31% and ASR at 6.72%.
5. **SCE-005 (Frontier Duel: LLM vs LLM)**: Evaluates generative reasoning in autonomous cyber warfare, achieving 49.80% detection, 12.69% ASR, and maintaining 100% service availability with the lowest intervention cost (346.40).

---

All telemetry, episode summaries, and hypothesis validations were recorded with live physical hardware in the loop (ESP8266 @ 192.168.0.111).