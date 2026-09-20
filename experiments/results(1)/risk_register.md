| Risk ID | Threat Description | STRIDE | Initial Risk | Control | Maturity | Residual Risk |
|:---|:---|:---|:---:|:---|:---:|:---:|
| RSK-01 | Unauthorized SSH brute-force credential stuffing | Spoofing | 16 (L4×I4) | Rate Limiting & Honeypot Redirection | 3.0/5.0 | **8.00** |
| RSK-02 | Remote command injection in web gateway | Elevation of Privilege | 20 (L4×I5) | Service Isolation & Input Sanitization | 2.5/5.0 | **11.67** |
| RSK-03 | IoT endpoint buffer overflow & OOM crash | Denial of Service | 15 (L5×I3) | Firmware Verification & IoT Isolation | 2.0/5.0 | **10.00** |
| RSK-04 | Lateral movement via compromised mesh nodes | Elevation of Privilege | 12 (L3×I4) | Micro-segmentation & Dynamic IP Blocking | 3.0/5.0 | **6.00** |
| RSK-05 | Stealthy C2 beaconing & data exfiltration | Information Disclosure | 15 (L3×I5) | Bayesian Anomaly Tracking & IDS Tuning | 2.5/5.0 | **8.75** |