# TAFFAC Threat Model and Adversary Profiling

## 1. Threat Assessment Framework (TAFFAC)

Adversary behaviour in **SecureMesh-SCE** is formally parameterized following the **TAFFAC framework** (Sood et al., 2026). Each adversary profile decouples the agent's observable policy from its latent operational capabilities.

### TAFFAC Attribute Taxonomy

| Attribute | Description | Levels / Domain |
|:---|:---|:---|
| **Capability** | Technical sophistication and toolset breadth | Low, Medium, High |
| **Access** | Initial foothold in network topology | None, Partial, Full |
| **System Knowledge** | Prior reconnaissance and architectural awareness | None, Partial, Full |
| **Objective** | Strategic primary mission | Disruption, Exfiltration, Full Compromise, Resource Hijack |
| **Attack Budget** | Maximum execution steps before exhaustion | 100 to 300 steps |
| **Adaptability** | Ability to pivot tactics when blocked | Deterministic (False), Dynamic (True) |
| **Autonomy** | Decision-making autonomy | Scripted, Semi-Autonomous, Autonomous RL |
| **Observability** | Information visibility over defensive responses | Blind, Partial, Full |

---

## 2. Hidden Adversary Types ($\Theta$)

SecureMesh-SCE models four canonical threat archetypes:

1. **Opportunistic ($\theta_1$)**:
   - *Profile*: Low capability, zero prior knowledge, high volume scanning, brute-forcing default credentials.
   - *Signature*: High scan and authentication attempt frequencies; predictable kill-chain progression; easily deterred by basic rate limiting.
2. **Stealth ($\theta_2$)**:
   - *Profile*: Medium capability, exfiltration focused, evasion-heavy, long inter-arrival delays.
   - *Signature*: Throttled traffic (`EVADE_SLOWDOWN`), obfuscated commands (`EVADE_OBFUSCATE`), low scan rates, persistent beaconing.
3. **Adaptive ($\theta_3$)**:
   - *Profile*: High capability, autonomous reinforcement learning (PPO), dynamic reward-seeking, exploits defensive blind spots.
   - *Signature*: Switches tactics immediately upon detection; alternates between IoT exploits and SSH compromise; probes honeypot thresholds.
4. **Resource-Aware ($\theta_4$)**:
   - *Profile*: High system knowledge, targets constrained IoT hardware (ESP8266 memory exhaustion, MQTT broker hijacking).
   - *Signature*: Rapid bursts targeting specific IoT ports to cause buffer overflows and watchdog brownouts.

---

## 3. STRIDE Classification and MITRE ATT&CK Mapping

Each attacker action maps directly to STRIDE categories and MITRE ATT&CK techniques:

| Attacker Action | STRIDE Category | MITRE ATT&CK Technique | Technique Name |
|:---|:---|:---:|:---|
| `RECON_SCAN` | Information Disclosure | **T1595.001** | Active Scanning: IP Blocks |
| `RECON_FINGERPRINT` | Information Disclosure | **T1592** | Gather Victim Host Information |
| `AUTH_BRUTEFORCE` | Spoofing | **T1110.001** | Password Guessing |
| `AUTH_CREDENTIAL_STUFF` | Spoofing | **T1110.004** | Credential Stuffing |
| `EXPLOIT_SERVICE` | Elevation of Privilege | **T1190** | Exploit Public-Facing Application |
| `EXPLOIT_IOT` | Tampering | **T1200** | Hardware Additions / Firmware Exploit |
| `MALWARE_DROP` | Tampering | **T1105** | Ingress Tool Transfer |
| `PERSIST_BACKDOOR` | Elevation of Privilege | **T1543** | Create or Modify System Process |
| `PERSIST_C2` | Information Disclosure | **T1071.001** | Web Protocols: C2 Beaconing |
| `EVADE_OBFUSCATE` | Repudiation | **T1027** | Obfuscated Files or Information |
| `EVADE_SLOWDOWN` | Repudiation | **T1029** | Scheduled / Throttled Transfer |
| `LATERAL_MOVE` | Elevation of Privilege | **T1021.004** | Remote Services: Lateral Movement |
