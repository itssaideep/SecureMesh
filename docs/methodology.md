# SCENE Experimental Methodology

## 1. Overview and Core Research Objective

**SecureMesh-SCE** is an AI-assisted Security Chaos Engineering (SCE) research platform engineered according to the **SCENE guidelines** (Jolak et al., 2026). It investigates the fundamental research question:

> *How does uncertainty about attacker capabilities affect the effectiveness and resilience of adaptive AI-based defence in an IoT security-chaos environment?*

In realistic IoT deployments, defenders operate under profound epistemic uncertainty: the adversary's capability, budget, persistence, and strategic adaptability are unobservable latent variables. Deterministic static defences and un-augmented reinforcement learning (RL) agents often either catastrophically over-react (causing denial-of-service via blind isolation) or fail to detect stealthy evasion tactics. SecureMesh-SCE formalises this interaction as a **two-player Bayesian Markov game**.

---

## 2. Theoretical Formulation: Bayesian Markov Game

The system is modelled as a game $\mathcal{G} = \langle \mathcal{S}, \mathcal{A}_A, \mathcal{A}_D, \Theta, \mathcal{T}, \mathcal{R}_A, \mathcal{R}_D, \gamma \rangle$:

1. **Adversary Type Space**: $\theta \in \Theta = \{\text{opportunistic}, \text{stealth}, \text{adaptive}, \text{resource\_aware}\}$.
   The true type $\theta^*$ is sampled at $t=0$ from a prior distribution $b_0(\theta)$ and remains constant throughout the episode. The defender cannot directly observe $\theta^*$.
2. **Attacker Action Space**: $\mathcal{A}_A$ comprises 13 kill-chain actions spanning Reconnaissance, Authentication, Exploitation, Persistence, Defense Evasion, and Lateral Movement.
3. **Defender Action Space**: $\mathcal{A}_D$ comprises 14 defensive responses classified into four Impact Tiers ($\text{Low}, \text{Medium}, \text{High}, \text{Critical}$) with reversibility annotations.
4. **State Representation**:
   - Attacker State: Physical network topology snapshot $x_t \in \mathbb{R}^{d_x}$.
   - Defender State: Augmented composite state $s_t = [x_t, b_t, h_t, r_t]$, where:
     - $x_t$: Raw network, service, and IoT device telemetry.
     - $b_t$: Exact Bayesian posterior belief distribution over $\Theta$.
     - $h_t$: Exponentially-decayed attack history summary vector over a sliding window.
     - $r_t$: Current risk and resilience state ($[\text{cumulative\_impact}, \text{service\_availability}, \text{recovery\_debt}, \text{security\_stage}]$).
5. **Exact Bayesian Belief Update**:
   $$b_t(\theta_i) = P(\theta_i \mid O_{0:t}) = \frac{P(O_t \mid \theta_i) b_{t-1}(\theta_i)}{\sum_{j=1}^{|\Theta|} P(O_t \mid \theta_j) b_{t-1}(\theta_j)}$$
   The defender preserves the full four-dimensional probability vector rather than collapsing to an argmax point estimate.
6. **Multi-Objective Defender Reward Function**:
   $$R_D = w_s \cdot \text{Security} + w_a \cdot \text{Availability} + w_r \cdot \text{Recovery} - w_c \cdot \text{DefenceCost} - w_i \cdot \text{UnnecessaryIntervention}$$
   This explicit multi-objective structure penalises the trivial failure mode of isolating the entire infrastructure immediately upon detecting suspicious activity.

---

## 3. The 8-Stage SCENE Chaos Lifecycle

Every scenario in SecureMesh-SCE follows the rigorous 8-stage SCENE experimental lifecycle:

```
[1. Define Hypothesis] ──► [2. Establish Steady State] ──► [3. Injected Chaos Fault]
                                                                     │
[6. Evaluate Hypothesis] ◄── [5. Anomaly Detection] ◄── [4. Joint Action Execution]
          │
          ▼
[7. Dynamic Risk Update] ──► [8. Export Telemetry & Paper Artifacts]
```

1. **Define Hypothesis**: Pre-registers a quantitative success criterion (e.g., metric, threshold, direction: `above` or `below`).
2. **Establish Steady State**: Validates operational baseline stability over a pre-chaos window (ensuring availability $\ge 0.99$ and zero active compromises).
3. **Select Scenario & Inject Chaos**: Injects adversary attacks or network chaos faults.
4. **Joint Action Execution**: The attacker agent $\pi_A(a_A \mid x_t)$ and defender agent $\pi_D(a_D \mid s_t)$ simultaneously select actions; the transition engine computes consequences.
5. **Telemetry & Anomaly Detection**: Logs latency, throughput, connection state, IDS alerts, and authentication failure spikes.
6. **Measure Outcome & Evaluate Hypothesis**: Performs multi-seed statistical significance testing (Mann-Whitney U, Cohen's $d$, 95% confidence intervals) to issue a formal `PASS` or `FAIL` verdict.
7. **Compute Residual Risk**: Dynamically updates control maturity and residual risk scores in the risk register.
8. **Export Telemetry & Artifacts**: Writes JSONL step logs, CSV episode records, and publication-ready LaTeX tables and vector figures.
