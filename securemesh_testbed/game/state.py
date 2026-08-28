# securemesh_testbed/game/state.py
"""Environment state representation.

The state is a dictionary returned by `NetworkSimulator.snapshot()` combined
with attacker‑specific fields. For the RL agents we flatten the state into a
numeric vector.
"""

from typing import Dict, Any
import numpy as np

def flatten_state(state: Dict[str, Any]) -> np.ndarray:
    """Convert the nested state dict into a 1‑D NumPy array.
    The ordering is deterministic so both agents can decode it.
    """
    parts = []
    # Hosts
    for hostname, hdata in sorted(state["hosts"].items()):
        parts.append(1.0 if hdata["isolated"] else 0.0)
        for svc_name, svc in sorted(hdata["services"].items()):
            parts.append(1.0 if svc["compromised"] else 0.0)
            parts.append(len(svc["active_sessions"]))
    # IoT devices
    for dev_id, ddata in sorted(state["iot_devices"].items()):
        parts.append(1.0 if ddata["isolated"] else 0.0)
        parts.append(1.0 if ddata["compromised"] else 0.0)
    return np.array(parts, dtype=np.float32)
