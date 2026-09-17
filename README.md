# SecureMesh-SCE

## A SCENE-Guided, AI-Assisted Security Chaos Engineering Testbed for Studying Adaptive Cyberattack and Cyberdefence in IoT Environments

## Introduction

SecureMesh-SCE is a controlled research testbed for studying adaptive cyberattack and cyberdefence in heterogeneous IoT environments.

The project extends an IoT security environment into a Security Chaos Engineering (SCE) platform. It combines controlled attack simulation, reinforcement learning, Bayesian reasoning, imitation learning, physical IoT devices, telemetry collection, and safety-constrained defensive actions.

The central research question is:

How does uncertainty about attacker capabilities affect the effectiveness and resilience of adaptive AI-based cyberdefence in an IoT security-chaos environment?

The project is designed as a reproducible research platform rather than only a honeypot or an intrusion-detection demonstration.

## Research Objectives

- Model interactions between adaptive attackers and adaptive defenders.
- Represent uncertainty about attacker capabilities using Bayesian beliefs.
- Study how attacker adaptation affects defensive resilience.
- Compare static, machine-learning, Bayesian, and constrained defence strategies.
- Use Security Chaos Engineering to create controlled and repeatable security disruptions.
- Evaluate AI-assisted defence in terms of security, availability, recovery, and intervention cost.
- Provide a reproducible environment for future research.

## Tech Stack

### Programming and AI
- Python
- PyTorch
- Reinforcement Learning
- Proximal Policy Optimization (PPO)
- Behavioral Cloning (BC)
- Generative Adversarial Imitation Learning (GAIL)
- Bayesian inference
- Random Forest
- Optional LLM integration

### IoT and Hardware
- ESP8266 NodeMCU
- ESP32
- Raspberry Pi
- Physical and simulated IoT endpoints

### Security and Networking
- Docker
- Cowrie / SSH honeypot
- HTTP services
- MQTT / Mosquitto
- IDS and telemetry components
- Network monitoring and logging

### Research and Evaluation
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Statistical analysis
- Experiment logging
- Multi-seed evaluation

## System Architecture

The system consists of an attacker layer, an environment layer, and a defender layer.

Attacker
    |
    v
+-----------------------------+
| SecureMesh-SCE Environment  |
|                             |
| IoT devices                 |
| Network services            |
| Telemetry                   |
| Security events             |
+-----------------------------+
    |
    v
Defender
    |
    v
Monitoring / Response / Recovery

The environment maintains system telemetry, attacker history, risk information, and the defender's belief about the hidden attacker type.

## Attacker Model

The project uses multiple attacker levels to study increasing levels of adaptability.

### L0 - Scripted Attacker
A deterministic attacker follows a predefined sequence of actions.

### L1 - Behavioral Cloning Attacker
The attacker learns from demonstrations using Behavioral Cloning.

### L2 - GAIL Attacker
Generative Adversarial Imitation Learning is used to learn behavior resembling demonstrated attacker behavior.

### L3 - PPO Attacker
The attacker uses Proximal Policy Optimization to learn an adaptive strategy through interaction with the environment.

### L4 - LLM Attacker
An optional LLM-based planner generates attacker decisions within the controlled testbed.

Possible attacker actions include reconnaissance, scanning, controlled credential attacks, web-exploit simulation, IoT exploit simulation, lateral-movement simulation, privilege-escalation simulation, C2 simulation, exfiltration simulation, log-clearing simulation, evasion simulation, and DoS simulation.

All attack behavior is intended for the isolated research environment and owned test devices.

## Bayesian Attacker Types

The defender does not directly observe the attacker's type.

Initial attacker types are:
- Opportunistic
- Stealth
- Adaptive
- Resource-aware

The attacker type represents hidden characteristics such as capabilities, objectives, knowledge, or strategic behavior.

The attacker type is different from the attacker's policy. Two attackers may have the same hidden type but use different policies, such as scripted behavior or PPO.

The defender maintains a probability distribution over possible attacker types and updates this belief as new observations arrive.

Conceptually:

P(attacker type | observations)

This allows the defender to represent uncertainty instead of making an immediate hard classification.

## Defender Model

### D1 - Static Defender
Uses predefined security rules and responses.

### D2 - Random Forest Defender
Uses a machine-learning classifier to support defensive decisions.

### D3 - RL Defender
Uses PPO to learn defensive actions from system state and rewards.

### D4 - Bayesian RL Defender
Combines Bayesian belief estimation with PPO-based decision making.

The Bayesian component estimates the likely attacker type, while PPO learns which defensive action to take.

### D5 - Constrained Defender
Adds a safety gate to the Bayesian RL defender.

The safety mechanism prevents high-impact actions from being executed unless the estimated risk and attacker belief satisfy predefined conditions.

## PPO Process

PPO stands for Proximal Policy Optimization.

The basic interaction loop is:

1. Observe the current system state.
2. Select an action.
3. Execute the action in the environment.
4. Observe the resulting state.
5. Receive a reward.
6. Estimate the advantage of the selected action.
7. Update the policy.
8. Repeat.

The "proximal" part of PPO refers to limiting how much the policy changes during an update, helping prevent unstable or excessively large policy updates.

In SecureMesh-SCE:
- Bayesian inference estimates uncertainty about the attacker.
- PPO learns defensive decisions.
- The environment provides the consequences of those decisions.
- The attacker can also adapt, allowing attacker-defender co-adaptation experiments.

## Imitation Learning

Imitation learning provides a progression from predefined behavior to adaptive behavior:

Scripted demonstrations
        |
        v
Behavioral Cloning
        |
        v
GAIL
        |
        v
PPO-based adaptive attacker

This makes it possible to study how increasing attacker adaptability changes defensive performance.

## Physical IoT Testbed

The ESP8266 is used as a physical, resource-constrained IoT target.

It provides real device telemetry and introduces practical constraints that are difficult to reproduce perfectly in simulation.

The ESP8266 can expose controlled research endpoints such as:
- A deceptive smart-plug administration interface
- A controlled login trap
- A command-injection simulation endpoint
- A status endpoint providing device telemetry

Example telemetry includes:
- Free heap memory
- Wi-Fi RSSI
- Uptime
- Service status
- Security events

The physical device is used as an IoT asset rather than as the location of Bayesian inference or the main AI logic. The intelligence remains in the security control plane.

## Safety Gate

The constrained defender uses different action tiers.

### Tier 1 - Low Impact
- NoAction
- Monitor
- IncreaseLogging

### Tier 2 - Medium Impact
- RateLimit
- HoneypotRedirect
- DeceiveAttacker
- RotateCredentials

### Tier 3 - High Impact
- BlockIP
- TerminateSession
- PatchVulnerability
- RestartService

### Tier 4 - Critical Impact
- IsolateIoTDevice
- SegmentNetwork
- RollbackState

The purpose of the safety gate is to prevent an AI policy from automatically taking high-impact actions when confidence or risk is insufficient.

## Security Chaos Engineering Process

SecureMesh-SCE uses Security Chaos Engineering as the experimental methodology.

The experimental process is:

1. Define a research hypothesis.
2. Establish a normal steady state.
3. Inject a controlled security disruption.
4. Allow attacker and defender actions.
5. Collect telemetry and security events.
6. Measure system impact and defensive response.
7. Analyse resilience and recovery.
8. Record findings and update the research risk register.

The experiments are designed to be repeatable so different defensive strategies can be compared under equivalent conditions.

## Experimental Scenarios

1. Scripted attacker / Static defender
2. Scripted attacker / RL defender
3. Behavioral Cloning attacker / Static defender
4. PPO attacker / Static defender
5. PPO attacker / RL defender
6. PPO attacker / Bayesian RL defender
7. PPO attacker / Constrained defender
8. LLM-assisted attacker / Bayesian RL defender

## Evaluation Metrics

### Security Performance
- Attack Success Rate (ASR)
- Detection Rate
- Precision
- Recall
- F1 Score
- False Positive Rate
- Mean Time to Detect (MTTD)
- Mean Time to Recover (MTTR)

### Defensive Performance
- Response latency
- Defence cost
- Unnecessary interventions
- Autonomous action rate
- High-impact action rate
- Human escalation rate
- Safety violations

### Bayesian Performance
- Attacker-type accuracy
- Brier Score
- Expected Calibration Error (ECE)
- Log Loss

### Resilience
- Service availability
- Performance degradation
- Cumulative impact
- Recovery time
- Recovery completeness

## Results

Results should be added only after the experiments have been executed.

The final evaluation should report results across multiple random seeds and preserve raw per-seed data.

Recommended reporting includes:
- Mean and median performance
- Standard deviation
- Confidence intervals
- Effect sizes
- Appropriate statistical significance tests
- Per-scenario results
- Ablation studies

Important comparisons include:
- Static vs adaptive attacker
- Static vs RL defence
- RL vs Bayesian RL defence
- Bayesian RL vs constrained defence
- Different levels of attacker adaptability
- Simulated vs physical IoT experiments

## Reproducibility

Experiments should use:
- Fixed random seeds where appropriate
- Versioned configurations
- Clearly defined scenarios
- Consistent action spaces
- Consistent telemetry features
- Raw experiment logs
- Per-seed results
- Reproducible environment configurations

Public datasets can inform feature and scenario construction, while controlled SecureMesh-SCE experiments generate the sequential attacker-defender interaction data required for evaluating adaptive defence.

## Research Data

### CICIoT2023
Used for IoT network traffic characterization, feature construction, scenario development, and baseline IDS validation.

### TON_IoT
Used to provide heterogeneous IoT/IIoT network, host, and telemetry data.

### SecureMesh-SCE Generated Dataset
The primary experimental dataset is generated by the SecureMesh-SCE environment.

This dataset captures sequential interactions between:

Attacker -> Environment -> Defender -> Environment -> Telemetry

This is important because the research focuses on adaptation and resilience rather than only classifying individual network events.

## Repository Structure

securemesh-sce/
|
+-- attackers/
|   +-- scripted/
|   +-- behavioral_cloning/
|   +-- gail/
|   +-- ppo/
|   +-- llm/
|
+-- defenders/
|   +-- static/
|   +-- random_forest/
|   +-- ppo/
|   +-- bayesian_rl/
|   +-- constrained/
|
+-- environment/
|   +-- simulation/
|   +-- services/
|   +-- telemetry/
|
+-- hardware/
|   +-- esp8266/
|   +-- esp32/
|
+-- experiments/
|   +-- scenarios/
|   +-- configurations/
|   +-- results/
|
+-- evaluation/
+-- data/
+-- docs/
+-- README.md

## Research Positioning

SecureMesh-SCE combines:
- Security Chaos Engineering
- Adaptive cyberattack and cyberdefence
- Bayesian decision making under uncertainty
- Reinforcement learning
- Imitation learning
- IoT security
- Safe and constrained autonomous defence
- Physical IoT experimentation

The intended contribution is not simply the use of PPO, Bayesian inference, SCENE, or LLMs individually.

The research contribution is the controlled investigation of how attacker uncertainty and attacker adaptation affect defensive resilience in an IoT Security Chaos Engineering environment.

## References

1. R. Jolak, R. R. Avula, M. Mohamad, "SCENE Guidelines and Live Systematic Literature Review for Security Chaos Engineering," Journal of Systems and Software, 2026.
2. A. K. Sood, S. Zeadally, E.-K. Hong, "TAFFAC: Threat Assessment Framework for Frontier AI Models for Cybersecurity," Computers & Electrical Engineering, vol. 139, 2026, Art. no. 111472.
3. Y. E. Smid, P. van der Putten, A. Plaat, "Mirror Mode in Fire Emblem: Beating Players at their Own Game with Imitation and Reinforcement Learning," Leiden University, 2025.
4. Y. Zhang, D. Goel, H. Ahmad, "Explainable Autonomous Cyber Defense using Adversarial Multi-Agent Reinforcement Learning," Expert Systems with Applications, vol. 326, 2026, Art. no. 132741.
5. K. Hammar, R. Stadler, "Finding Effective Security Strategies through Reinforcement Learning and Self-Play," IEEE/IFIP CNSM, 2020.
6. H. Alavizadeh, J. Jang-Jaccard, T. Alpcan, S. A. Camtepe, "A Markov Game Model for AI-based Cyber Security Attack Mitigation," arXiv:2107.09258, 2021.
7. W. Kalka, T. Szydlo, "Î¼Chaos: Moving Chaos Engineering to IoT Devices," ICCS, Springer, 2024.
8. B. Basiri et al., "Chaos Engineering," IEEE Internet Computing, 2016.
9. K. A. Torkura et al., "CloudStrike: Chaos engineering for security and resiliency in cloud infrastructure," IEEE Access, 2020.
10. K. A. Torkura et al., "Continuous auditing and threat detection in multi-cloud infrastructure," Computers & Security, 2021.

## Disclaimer

SecureMesh-SCE is intended for controlled cybersecurity research and experimentation on isolated environments and owned devices. Attack behaviors should not be deployed against systems without explicit authorization.

