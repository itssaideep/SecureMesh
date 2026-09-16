# Threats to Validity and Simulation Boundaries

A rigorous scientific platform must document its assumptions, boundary conditions, and threats to validity.

---

## 1. Threats to Validity

### Internal Validity
1. **Likelihood Prior Specification**:
   - The Bayesian inference module uses pre-calibrated conditional distributions $P(a_A \mid \theta_i)$ derived from observed kill-chain distributions. While effective for known threat profiles, zero-day behaviors that deviate significantly from all four prior models could initially degrade posterior calibration until updated.
2. **Markovian State Assumption**:
   - The simulation abstracts physical state transitions into discrete steps. Although an exponential sliding-window history vector $h_t$ is incorporated into defender observations to mitigate non-Markovian dynamics, real-world low-rate packet manipulation operates at sub-millisecond granularity.
3. **Simulated Hardware Constraints**:
   - The software simulators for ESP8266 and ESP32 faithfully model memory consumption (heap depletion, watchdog brownouts, crypto acceleration latency), but do not replicate exact hardware-level electromagnetic side-channel phenomena.

### External Validity
1. **Protocol Generalizability**:
   - Current simulated services emphasize SSH, HTTP, and MQTT—the predominant management and telemetry protocols in contemporary commercial IoT. While the Bayesian Markov game formulation is protocol-agnostic, validating on industrial SCADA protocols (e.g., Modbus, DNP3) remains future work.
2. **Topology Scale**:
   - The default scenario represents a heterogeneous edge subnet (gateway, workstations, honeypot nodes, and constrained sensor endpoints). While representative of edge IoT deployments, enterprise-scale environments with thousands of nodes require hierarchical sub-game decomposition.

### Construct Validity
1. **Multi-Objective Weight Balancing**:
   - The defender reward weights ($w_s, w_a, w_r, w_c, w_i$) reflect practical engineering trade-offs between availability and containment. Sensitivity analyses demonstrate that findings are robust across proportional weight variations, but specific deployments may place higher utility on absolute confidentiality versus uptime.
2. **Autonomous Gating Boundaries**:
   - The safety filter employs deterministic contract gating based on confidence entropy and security stage. While this strictly prevents unauthorized service outages, highly novel evasive attacks could exploit conservative gating delays.

---

## 2. Simulation Boundaries & Scope

| Dimension | In-Scope | Out-of-Scope |
|:---|:---|:---|
| **Adversary Modeling** | Scripted, Imitation (BC/GAIL), Adaptive PPO, Generative LLM | Distributed multi-agent botnet DDoS (>10,000 bots) |
| **Defensive Architecture** | Static, ML, RL, Bayesian RL, Constrained Safety Gates | Hardware physically unpluggable air-gap switches |
| **Inference Framework** | Full Bayesian posterior tracking $b_t(\theta) \in \Delta^4$ | Particle filters over infinite continuous parameter spaces |
| **Evaluation Metrics** | ASR, DR, MTTD, MTTR, Brier, ECE, Availability, Cost | Financial loss dollar-value modeling |
