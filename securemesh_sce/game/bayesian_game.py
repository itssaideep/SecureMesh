# securemesh_sce/game/bayesian_game.py
"""Gymnasium-compatible two-player Markov Game environment.

The central environment for SecureMesh-SCE. Models the interaction
between an attacker and a defender as a two-player simultaneous-action
Markov game — the game-theoretic formulation for the thesis.

Key design decisions:
- The attacker type θ is sampled at episode start and remains fixed.
- Both agents observe the network system state and choose actions simultaneously.
- Service availability trades off with security interventions.
- The environment supports both action-index and action-enum interfaces.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .actions import (
    AttackerAction, DefenderAction, AttackerType, AttackerProfile,
    N_ATTACKER_ACTIONS, N_DEFENDER_ACTIONS, N_ATTACKER_TYPES,
    ATTACK_IMPACT_WEIGHTS, SecurityStage,
)
from .state import (
    flatten_network_state, AttackHistory, RiskState, DefenderState,
)
from .rewards import (
    attacker_reward, defender_reward, StepOutcome,
    AttackerRewardWeights, DefenderRewardWeights,
    DEFENDER_ACTION_COSTS,
)
from .transitions import TransitionEngine, ATTACKER_LIKELIHOOD_TABLE


# =====================================================================
# Scenario configuration
# =====================================================================

@dataclass
class SCEScenario:
    """Configuration for a single experiment scenario."""
    # Identity
    scenario_id: str = "SCE-001"
    name: str = "Baseline"
    domain: str = "IoT"

    # Network topology
    n_hosts: int = 3
    services_per_host: int = 2
    n_iot_devices: int = 2
    base_vulnerability: float = 0.3

    # Attacker
    attacker_type: Optional[AttackerType] = None  # None = random
    attacker_budget: int = 200  # max steps before exhaustion

    # Environment
    duration: int = 100  # steps per episode
    seed: int = 42

    # IDS
    ids_base_sensitivity: float = 0.5

    # Reward weights
    attacker_reward_weights: AttackerRewardWeights = field(
        default_factory=AttackerRewardWeights
    )
    defender_reward_weights: DefenderRewardWeights = field(
        default_factory=DefenderRewardWeights
    )


# =====================================================================
# Two-player Markov Game environment
# =====================================================================

class BayesianGameEnv(gym.Env):
    """Two-player Markov Game for AI Cyberattack vs Cyberdefence.

    Observation spaces:
        - Attacker: flat system state (full observability of network)
        - Defender: [system_state, history, risk]

    Action spaces:
        - Attacker: Discrete(N_ATTACKER_ACTIONS)
        - Defender: Discrete(N_DEFENDER_ACTIONS)

    The game returns a tuple:
        (atk_obs, def_obs), (atk_reward, def_reward), done, truncated, info
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario: Optional[SCEScenario] = None):
        super().__init__()
        self.scenario = scenario or SCEScenario()

        # Random state
        self._seed = self.scenario.seed
        self.rng = random.Random(self._seed)
        self.np_rng = np.random.RandomState(self._seed)

        # Transition engine
        self.transition_engine = TransitionEngine(rng=self.rng)

        # State components
        self.attack_history = AttackHistory()
        self.risk_state = RiskState()

        # Network state (initialised in reset)
        self._network_state: Dict[str, Any] = {}
        self._ids_sensitivity = self.scenario.ids_base_sensitivity
        self._attacker_type: AttackerType = AttackerType.OPPORTUNISTIC
        self._attacker_type_idx: int = 0
        self._security_stage = SecurityStage.NORMAL

        # Step tracking
        self.current_step = 0
        self.done = False
        self.max_steps = self.scenario.duration

        # Telemetry
        self._step_log: List[Dict[str, Any]] = []
        self._episode_outcomes: List[StepOutcome] = []

        # Action and observation spaces
        self.attacker_action_space = spaces.Discrete(N_ATTACKER_ACTIONS)
        self.defender_action_space = spaces.Discrete(N_DEFENDER_ACTIONS)

        # Observation dims (computed after first reset)
        self._atk_obs_dim: int = 1
        self._def_obs_dim: int = 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict]:
        """Reset the environment for a new episode.

        Returns
        -------
        (atk_obs, def_obs) : tuple of np.ndarray
        info : dict
        """
        if seed is not None:
            self._seed = seed
            self.rng = random.Random(seed)
            self.np_rng = np.random.RandomState(seed)
            self.transition_engine = TransitionEngine(rng=self.rng)

        # Sample attacker type
        if self.scenario.attacker_type is not None:
            self._attacker_type = self.scenario.attacker_type
        else:
            self._attacker_type = self.rng.choice(list(AttackerType))
        self._attacker_type_idx = list(AttackerType).index(self._attacker_type)

        # Build network state
        self._network_state = self._build_network()
        self._ids_sensitivity = self.scenario.ids_base_sensitivity
        self._security_stage = SecurityStage.NORMAL

        # Reset components
        self.attack_history.reset()
        self.risk_state.reset()

        self.current_step = 0
        self.done = False
        self._step_log = []
        self._episode_outcomes = []

        # Build observations
        atk_obs = self._attacker_observation()
        def_obs = self._defender_observation()
        self._atk_obs_dim = len(atk_obs)
        self._def_obs_dim = len(def_obs)

        info = {
            "attacker_type": self._attacker_type.value,
            "attacker_type_idx": self._attacker_type_idx,
        }
        return (atk_obs, def_obs), info

    # ==================================================================
    # Step
    # ==================================================================

    def step(
        self,
        actions: Tuple[int, int],
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[float, float],
        bool,
        bool,
        Dict[str, Any],
    ]:
        """Take a joint attacker/defender step.

        Parameters
        ----------
        actions : (attacker_action_idx, defender_action_idx)

        Returns
        -------
        (atk_obs, def_obs), (atk_r, def_r), terminated, truncated, info
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        atk_idx, def_idx = actions
        atk_action = list(AttackerAction)[atk_idx % N_ATTACKER_ACTIONS]
        def_action = list(DefenderAction)[def_idx % N_DEFENDER_ACTIONS]

        # --- Resolve attacker action ---
        outcome = self.transition_engine.resolve_attacker(
            action=atk_action,
            attacker_type=self._attacker_type,
            network_state=self._network_state,
            ids_sensitivity=self._ids_sensitivity,
        )

        # --- Resolve defender action ---
        outcome = self.transition_engine.resolve_defender(
            action=def_action,
            attacker_outcome=outcome,
            network_state=self._network_state,
        )

        # --- Update network state ---
        self._apply_state_changes(atk_action, def_action, outcome)

        # --- Update attack history ---
        history_features = self._extract_history_features(atk_action, outcome)
        self.attack_history.update(history_features)

        # --- Update risk state ---
        self._update_risk_state(outcome)

        # --- Compute rewards ---
        atk_r = attacker_reward(outcome, self.scenario.attacker_reward_weights)
        def_r = defender_reward(outcome, self.scenario.defender_reward_weights)

        # --- Advance step ---
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        self.done = terminated

        # --- Build observations ---
        atk_obs = self._attacker_observation()
        def_obs = self._defender_observation()

        # --- Build info ---
        info = {
            "step": self.current_step,
            "attacker_action": atk_action.name,
            "defender_action": def_action.name,
            "attacker_type": self._attacker_type.value,
            "attack_success": outcome.attack_success,
            "attacker_detected": outcome.attacker_detected,
            "correct_detection": outcome.correct_detection,
            "false_positive": outcome.false_positive,
            "missed_attack": outcome.missed_attack,
            "service_down": outcome.service_down,
            "service_restored": outcome.service_restored,
            "impact_score": outcome.impact_score,
            "security_stage": self._security_stage.name,
            "service_availability": self.risk_state.service_availability,
            "cumulative_impact": self.risk_state.cumulative_impact,
        }

        self._step_log.append(info)
        self._episode_outcomes.append(outcome)

        return (atk_obs, def_obs), (atk_r, def_r), terminated, truncated, info

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _build_network(self) -> Dict[str, Any]:
        """Construct the initial network state dict."""
        hosts = {}
        for i in range(self.scenario.n_hosts):
            hostname = f"host_{i}"
            services = {}
            for j in range(self.scenario.services_per_host):
                svc_name = f"svc_{i}_{j}"
                vuln = self.scenario.base_vulnerability + self.rng.gauss(0, 0.05)
                vuln = max(0.05, min(0.95, vuln))
                services[svc_name] = {
                    "compromised": False,
                    "active_sessions": 0,
                    "port": 22 + j * 80,
                    "vulnerability": vuln,
                }
            hosts[hostname] = {"isolated": False, "services": services}

        iot_devices = {}
        for k in range(self.scenario.n_iot_devices):
            dev_id = f"iot_{k}"
            iot_devices[dev_id] = {
                "isolated": False,
                "compromised": False,
                "firmware_integrity": 1.0,
            }

        return {"hosts": hosts, "iot_devices": iot_devices}

    def _apply_state_changes(
        self,
        atk_action: AttackerAction,
        def_action: DefenderAction,
        outcome: StepOutcome,
    ):
        """Mutate network state based on step outcome."""
        # Attacker compromises
        if outcome.attack_success:
            if atk_action in (AttackerAction.EXPLOIT_SERVICE, AttackerAction.AUTH_BRUTEFORCE,
                              AttackerAction.AUTH_CREDENTIAL_STUFF, AttackerAction.MALWARE_DROP,
                              AttackerAction.PERSIST_BACKDOOR, AttackerAction.PERSIST_C2):
                # Pick a random non-isolated host/service and mark compromised
                available = [
                    (h, s)
                    for h, hd in self._network_state["hosts"].items()
                    if not hd["isolated"]
                    for s in hd["services"]
                ]
                if available:
                    h, s = self.rng.choice(available)
                    self._network_state["hosts"][h]["services"][s]["compromised"] = True
                    self._network_state["hosts"][h]["services"][s]["active_sessions"] += 1

            elif atk_action in (AttackerAction.EXPLOIT_IOT,):
                available = [
                    d for d, dd in self._network_state["iot_devices"].items()
                    if not dd["isolated"] and not dd["compromised"]
                ]
                if available:
                    d = self.rng.choice(available)
                    self._network_state["iot_devices"][d]["compromised"] = True

        # Defender isolations
        if def_action == DefenderAction.ISOLATE_SERVICE:
            hosts = [h for h, hd in self._network_state["hosts"].items()
                     if not hd["isolated"]]
            if hosts:
                h = self.rng.choice(hosts)
                self._network_state["hosts"][h]["isolated"] = True
                for svc in self._network_state["hosts"][h]["services"].values():
                    svc["compromised"] = False
                    svc["active_sessions"] = 0

        elif def_action == DefenderAction.ISOLATE_IOT:
            devs = [d for d, dd in self._network_state["iot_devices"].items()
                    if not dd["isolated"]]
            if devs:
                d = self.rng.choice(devs)
                self._network_state["iot_devices"][d]["isolated"] = True
                self._network_state["iot_devices"][d]["compromised"] = False

        elif def_action == DefenderAction.RESTORE_SERVICE:
            # Restore one isolated host
            isolated = [h for h, hd in self._network_state["hosts"].items()
                        if hd["isolated"]]
            if isolated:
                h = self.rng.choice(isolated)
                self._network_state["hosts"][h]["isolated"] = False

        elif def_action == DefenderAction.TERMINATE_SESSION:
            for hd in self._network_state["hosts"].values():
                for svc in hd["services"].values():
                    svc["active_sessions"] = 0

        elif def_action == DefenderAction.RESET_CREDENTIALS:
            for hd in self._network_state["hosts"].values():
                for svc in hd["services"].values():
                    if svc["compromised"]:
                        svc["compromised"] = False

        elif def_action == DefenderAction.BLOCK_IP:
            if outcome.attacker_blocked:
                for hd in self._network_state["hosts"].values():
                    for svc in hd["services"].values():
                        svc["active_sessions"] = 0

        elif def_action == DefenderAction.INCREASE_MONITORING:
            self._ids_sensitivity = min(1.0, self._ids_sensitivity + 0.05)

        elif def_action == DefenderAction.ADJUST_IDS_THRESHOLD:
            self._ids_sensitivity = min(1.0, self._ids_sensitivity + 0.03)

        elif def_action == DefenderAction.NETWORK_SHUTDOWN:
            for hd in self._network_state["hosts"].values():
                hd["isolated"] = True
                for svc in hd["services"].values():
                    svc["compromised"] = False
                    svc["active_sessions"] = 0
            for dd in self._network_state["iot_devices"].values():
                dd["isolated"] = True
                dd["compromised"] = False

    def _update_risk_state(self, outcome: StepOutcome):
        """Update the risk state based on step outcome."""
        if outcome.impact_score > 0:
            self.risk_state.cumulative_impact += outcome.impact_score

        if outcome.service_down:
            self.risk_state.recovery_debt += 1.0

        if outcome.service_restored:
            self.risk_state.recovery_debt = max(0, self.risk_state.recovery_debt - 1.0)

        # Compute service availability
        total_services = 0
        available_services = 0
        for hd in self._network_state["hosts"].values():
            for svc in hd["services"].values():
                total_services += 1
                if not hd.get("isolated", False):
                    available_services += 1
        for dd in self._network_state["iot_devices"].values():
            total_services += 1
            if not dd.get("isolated", False):
                available_services += 1

        self.risk_state.service_availability = (
            available_services / total_services if total_services > 0 else 1.0
        )

        # Update security stage based on threat level
        if outcome.attack_success and not outcome.correct_detection:
            self._security_stage = SecurityStage.SUSPICIOUS
        elif outcome.correct_detection:
            self._security_stage = SecurityStage.CONFIRMED
        elif outcome.service_restored:
            self._security_stage = SecurityStage.RECOVER

        self.risk_state.security_stage = self._security_stage.value

    def _extract_history_features(
        self, action: AttackerAction, outcome: StepOutcome
    ) -> Dict[str, float]:
        """Extract observation features for the attack history tracker."""
        features: Dict[str, float] = {}
        features["scan_rate"] = 1.0 if action in (
            AttackerAction.RECON_SCAN, AttackerAction.RECON_FINGERPRINT
        ) else 0.0
        features["auth_attempts"] = 1.0 if action in (
            AttackerAction.AUTH_BRUTEFORCE, AttackerAction.AUTH_CREDENTIAL_STUFF
        ) else 0.0
        features["exploit_attempts"] = 1.0 if action in (
            AttackerAction.EXPLOIT_SERVICE, AttackerAction.EXPLOIT_IOT
        ) else 0.0
        features["malware_attempts"] = 1.0 if action == AttackerAction.MALWARE_DROP else 0.0
        features["persistence_attempts"] = 1.0 if action in (
            AttackerAction.PERSIST_BACKDOOR, AttackerAction.PERSIST_C2
        ) else 0.0
        features["evasion_actions"] = 1.0 if action in (
            AttackerAction.EVADE_OBFUSCATE, AttackerAction.EVADE_SLOWDOWN
        ) else 0.0
        features["lateral_moves"] = 1.0 if action == AttackerAction.LATERAL_MOVE else 0.0
        features["connection_rate"] = 1.0 if outcome.attack_success else 0.0
        features["unique_targets"] = 0.0  # simplified
        features["noop_rate"] = 1.0 if action == AttackerAction.NOOP else 0.0
        return features

    def _attacker_observation(self) -> np.ndarray:
        """Build attacker observation (full network state)."""
        return flatten_network_state(self._network_state)

    def _defender_observation(self) -> np.ndarray:
        """Build defender observation: [x_t, h_t, r_t]."""
        x_t = flatten_network_state(self._network_state)
        h_t = self.attack_history.vector()
        r_t = self.risk_state.vector()
        return np.concatenate([x_t, h_t, r_t])

    # ==================================================================
    # Query methods
    # ==================================================================

    @property
    def attacker_type(self) -> AttackerType:
        return self._attacker_type

    @property
    def attacker_type_idx(self) -> int:
        return self._attacker_type_idx

    @property
    def step_log(self) -> List[Dict[str, Any]]:
        return self._step_log

    @property
    def episode_outcomes(self) -> List[StepOutcome]:
        return self._episode_outcomes
