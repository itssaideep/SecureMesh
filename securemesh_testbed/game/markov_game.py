# securemesh_testbed/game/markov_game.py
"""Gymnasium compatible two‑player Markov game.

The environment expects the attacker to act first, then the defender. It
returns a tuple of observations, rewards, done flag and an info dict.
Both agents receive their own partial observation.
"""

import random
from typing import Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces

from .actions import AttackerAction, DefenderAction
from .state import flatten_state
from ..environment.network_sim import NetworkSimulator, Service, Host, IoTDevice
from ..environment.service_sim import SSHService, HTTPService
from ..environment.ids_engine import IDSEngine
from .rewards import attacker_reward, defender_reward


class SecureMeshEnv(gym.Env):
    """Two‑player environment for the SecureMesh testbed.

    Observation spaces are `Dict` with a single key `obs` containing a
    1‑D float array (the flattened state). Action spaces are `Discrete`
    with size equal to the number of enum members.
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario):
        super().__init__()
        self.scenario = scenario
        # Build network based on scenario configuration
        hosts = []
        for svc_spec in scenario.services:
            if svc_spec.name == "ssh":
                svc = SSHService(port=svc_spec.port, vulnerability=svc_spec.vulnerability)
            elif svc_spec.name == "http":
                svc = HTTPService(port=svc_spec.port, vulnerability=svc_spec.vulnerability)
            else:
                svc = Service(name=svc_spec.name, port=svc_spec.port, vulnerability=svc_spec.vulnerability)
            host = Host(hostname=svc_spec.name, services=[svc])
            hosts.append(host)
        iot_devices = []
        for iot in scenario.iot_devices:
            iot_devices.append(IoTDevice(device_id=iot.device_id, firmware_hash=iot.firmware_hash, vulnerable=iot.vulnerable))
        self.network = NetworkSimulator(hosts=hosts, iot_devices=iot_devices)
        self.ids = IDSEngine(signature_mode=scenario.ids_signature_mode)

        # Action spaces
        self.attacker_action_space = spaces.Discrete(len(AttackerAction))
        self.defender_action_space = spaces.Discrete(len(DefenderAction))
        # Observation space – we will lazily infer size after first reset
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=float)

        self.max_steps = 100
        self.current_step = 0
        self.done = False
        random.seed(scenario.seed)

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, Any], Dict]:
        if seed is not None:
            random.seed(seed)
        self.network.reset()
        self.ids.reset()
        self.current_step = 0
        self.done = False
        raw_state = self.network.snapshot()
        flat = flatten_state(raw_state)
        # update observation space shape once we know length
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=flat.shape, dtype=float)
        attacker_obs = {"obs": flat.copy()}
        defender_obs = {"obs": flat.copy()}
        return attacker_obs, defender_obs

    # Helper methods to apply actions
    def _apply_attacker(self, action: AttackerAction) -> Tuple[bool, bool, bool]:
        """Execute attacker action.

        Returns a tuple (success, detected, blocked) used for reward.
        """
        # For simplicity we use a very lightweight logic
        success = False
        detected = False
        blocked = False
        # Choose a random host/service for actions that need a target
        host = random.choice(list(self.network.hosts.values()))
        svc = random.choice(list(host.services.values()))
        if action == AttackerAction.RECON_SCAN:
            # Scanning always succeeds; detection probability depends on IDS configuration
            detected = self.ids.check_scan()
        elif action == AttackerAction.AUTH_BRUTEFORCE:
            if isinstance(svc, SSHService):
                success = svc.auth_attempt("admin", "password", "attacker")
                detected = self.ids.check_auth(svc.failed_attempts.get("attacker", 0))
                blocked = self.ids.is_ip_blocked("attacker")
        elif action == AttackerAction.EXPLOIT_SERVICE:
            success = svc.attempt_exploit("attacker")
            detected = self.ids.check_exploit(success)
        elif action == AttackerAction.MALWARE_DROP:
            if isinstance(svc, HTTPService):
                success = svc.upload_file("malware.bin", b"\x00"*10, "attacker")
                detected = self.ids.check_malware(success)
        elif action == AttackerAction.PERSIST_BACKDOOR:
            # treat as success if any service already compromised
            success = any(s.compromised for s in host.services.values())
            detected = self.ids.check_persistence(success)
        elif action == AttackerAction.NOOP:
            pass
        # Other actions can be added later
        return success, detected, blocked

    def _apply_defender(self, action: DefenderAction,
                        atk_success: bool) -> Tuple[bool, bool, bool, bool]:
        """Execute defender action.

        Returns (detected, false_positive, service_down, service_restored)
        used for reward and metrics.
        """
        detected = False
        false_positive = False
        service_down = False
        service_restored = False

        # Determine if a real threat exists right now
        any_compromised = any(
            s.compromised for h in self.network.hosts.values()
            for s in h.services.values()
        )
        any_active_sessions = any(
            "attacker" in s.active_sessions
            for h in self.network.hosts.values()
            for s in h.services.values()
        )
        real_threat = any_compromised or any_active_sessions or atk_success

        # Simple rule‑based implementation
        if action == DefenderAction.BLOCK_IP:
            self.ids.block_ip("attacker")
            if real_threat:
                detected = True
            else:
                false_positive = True
        elif action == DefenderAction.RATE_LIMIT:
            self.ids.rate_limit_ip("attacker")
            if real_threat:
                detected = True
            else:
                false_positive = True
        elif action == DefenderAction.ISOLATE_SERVICE:
            host = random.choice(list(self.network.hosts.values()))
            svc = random.choice(list(host.services.values()))
            host.isolate()
            service_down = True
            if svc.compromised:
                detected = True
                svc.compromised = False
                svc.active_sessions.clear()
                service_restored = True
            elif not real_threat:
                false_positive = True
        elif action == DefenderAction.INCREASE_MONITORING:
            self.ids.increase_sensitivity()
        elif action == DefenderAction.ADJUST_IDS_THRESHOLD:
            self.ids.adjust_threshold(0.1)  # tighten
        elif action == DefenderAction.TERMINATE_SESSION:
            had_sessions = False
            for h in self.network.hosts.values():
                for s in h.services.values():
                    if "attacker" in s.active_sessions:
                        s.active_sessions.remove("attacker")
                        had_sessions = True
            if had_sessions:
                detected = True
                service_restored = True
            elif not real_threat:
                false_positive = True
        elif action == DefenderAction.NOOP:
            pass
        return detected, false_positive, service_down, service_restored

    def step(self, actions: Tuple[int, int]):
        """Take a joint attacker/defender step.

        `actions` – (attacker_action_index, defender_action_index)
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset() first.")
        attacker_idx, defender_idx = actions
        _atk_members = list(AttackerAction)
        _def_members = list(DefenderAction)
        attacker_action = _atk_members[attacker_idx % len(_atk_members)]
        defender_action = _def_members[defender_idx % len(_def_members)]
        # Apply attacker first
        atk_success, atk_detected, atk_blocked = self._apply_attacker(attacker_action)
        # Apply defender second
        def_detected, def_false_positive, def_service_down, def_service_restored = \
            self._apply_defender(defender_action, atk_success)

        # Compute impact score based on attack severity
        impact_score = 0.0
        if atk_success:
            _impact_weights = {
                AttackerAction.RECON_SCAN: 0.05,
                AttackerAction.AUTH_BRUTEFORCE: 0.3,
                AttackerAction.EXPLOIT_SERVICE: 0.6,
                AttackerAction.MALWARE_DROP: 0.8,
                AttackerAction.PERSIST_BACKDOOR: 1.0,
            }
            impact_score = _impact_weights.get(attacker_action, 0.1)

        # Determine missed attack and attacker isolation
        missed_attack = atk_success and not atk_detected and not def_detected
        attacker_isolated = (
            defender_action == DefenderAction.BLOCK_IP
            or defender_action == DefenderAction.ISOLATE_SERVICE
        ) and def_detected

        # Compute rewards (SCE profile by default)
        attacker_r = attacker_reward(
            atk_success, atk_detected, atk_blocked,
            recon_value=(attacker_action == AttackerAction.RECON_SCAN),
            isolated=attacker_isolated,
        )
        defender_r = defender_reward(
            atk_detected or def_detected, def_false_positive, def_service_down,
            isolated_attacker=attacker_isolated,
            missed_attack=missed_attack,
            service_maintained=not def_service_down,
        )

        # Update step counter
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        # Build observations for next step
        raw_state = self.network.snapshot()
        flat = flatten_state(raw_state)
        attacker_obs = {"obs": flat.copy()}
        defender_obs = {"obs": flat.copy()}
        info = {
            "attacker_success": atk_success,
            "attacker_detected": atk_detected,
            "defender_detected": def_detected,
            "false_positive": def_false_positive,
            "service_down": def_service_down,
            "service_restored": def_service_restored,
            "impact_score": impact_score,
        }
        return (attacker_obs, defender_obs), (attacker_r, defender_r), self.done, info

    def render(self, mode="human"):
        # Simple textual representation
        state = self.network.snapshot()
        print(state)

    def close(self):
        pass
