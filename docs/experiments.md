# Experimental Design and Canonical Scenarios

SecureMesh-SCE defines eight canonical experiment scenarios spanning deterministic baselines, imitation learning, co-adaptation, Bayesian uncertainty reasoning, and safety-constrained autonomy.

---

## 1. Canonical Scenarios (Experiments 1–8)

### Experiment 1: Deterministic Baseline (`01_baseline.yaml`)
- **Attacker**: Level 0 Scripted (`scripted`)
- **Defender**: Level 1 Static Rule-Based (`static`)
- **Hypothesis**: The static defender detects predictable scripted kill-chains with at least 50% detection rate without compromising service availability.
- **Objective**: Establish baseline benchmark for detection latency and steady-state availability.

### Experiment 2: RL Defense vs Scripted Attacker (`02_defense_rl.yaml`)
- **Attacker**: Level 0 Scripted (`scripted`)
- **Defender**: Level 3 RL Defender (`rl`)
- **Hypothesis**: The RL defender achieves higher detection rate ($\ge 0.60$) and lower cumulative impact than static rules.
- **Objective**: Verify whether un-augmented RL can learn effective defensive thresholds from raw system telemetry $x_t$.

### Experiment 3: Imitation Learning Offensive Policy (`03_imitation.yaml`)
- **Attacker**: Level 1 Behavioral Cloning (`bc`)
- **Defender**: Level 1 Static (`static`)
- **Hypothesis**: The imitation attacker achieves an attack success rate (ASR) of at least 25% by exploiting rule blind spots.
- **Objective**: Study how well supervised policies mimic human/advanced trajectories to bypass naive heuristics.

### Experiment 4: Attacker Adaptation via PPO (`04_attacker_adaptation.yaml`)
- **Attacker**: Level 3 Adaptive PPO (`ppo`)
- **Defender**: Level 1 Static (`static`)
- **Hypothesis**: The adaptive PPO attacker discovers vulnerabilities in deterministic defenses, achieving ASR $\ge 0.35$.
- **Objective**: Demonstrate the vulnerability of static defenses against reward-maximizing adversaries.

### Experiment 5: Adversarial Co-adaptation Dynamics (`05_co_adaptation.yaml`)
- **Attacker**: Level 3 Adaptive PPO (`ppo`)
- **Defender**: Level 3 RL Defender (`rl`)
- **Hypothesis**: Simultaneous co-adaptation maintains service availability above 0.75 despite continuous offensive strategy exploration.
- **Objective**: Measure policy stability, cycling, and equilibrium dynamics in un-augmented two-player RL.

### Experiment 6: Bayesian Belief Tracking Advantage (`06_bayesian_defense.yaml`)
- **Attacker**: Level 3 Adaptive PPO (`ppo`)
- **Defender**: Level 4 Bayesian RL (`bayesian_rl`)
- **Hypothesis**: Explicit Bayesian posterior belief tracking reduces ASR below 0.30 while increasing availability.
- **Objective**: Directly answer the core research question: quantify the resilience gain from tracking latent adversary uncertainty.

### Experiment 7: Bounded Autonomy and Action Gating (`07_constrained_safety.yaml`)
- **Attacker**: Level 3 Adaptive PPO (`ppo`)
- **Defender**: Level 5 Constrained Defender (`constrained`)
- **Hypothesis**: Safety-gated Bayesian RL preserves service availability above 0.85 and eliminates unauthorized critical interventions.
- **Objective**: Validate the staged escalation contract (`NORMAL` $\to$ `CONFIRMED` $\to$ `MITIGATE`) in preventing self-inflicted outages.

### Experiment 8: LLM Offensive Strategy (`08_llm_offensive.yaml`)
- **Attacker**: Level 4 LLM Agent (`llm` via Ollama/Gemini)
- **Defender**: Level 4 Bayesian RL (`bayesian_rl`)
- **Hypothesis**: The Bayesian defender detects and mitigates LLM-generated attack sequences with at least 65% detection rate.
- **Objective**: Explore the boundary of generative cyber offensive agents against Bayesian inference defenders.

---

## 2. Matrix and Ablation Study Design

The evaluation grid executes a complete $5 \times 5$ cross-product across 3 to 5 independent seeds:

| Attacker Level | Static (D1) | ML (D2) | RL (D3) | Bayesian RL (D4) | Constrained (D5) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Scripted (L0)** | Exp 1 | Baseline ML | Exp 2 | Ablation | Ablation |
| **BC Imitation (L1)**| Exp 3 | Supervised | RL vs BC | Ablation | Ablation |
| **GAIL (L2)** | GAIL vs Static | GAIL vs ML | GAIL vs RL | Ablation | Ablation |
| **Adaptive PPO (L3)**| Exp 4 | PPO vs ML | Exp 5 | Exp 6 | Exp 7 |
| **LLM-Assisted (L4)**| LLM vs Static | LLM vs ML | LLM vs RL | Exp 8 | Ablation |
