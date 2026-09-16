# securemesh_sce/agents/common/mlp.py
"""NumPy-only Multi-Layer Perceptron with analytical backpropagation.

Supports configurable depth, width, and activation functions.
Used by all RL-based agents (PPO, BC, GAIL).
"""

from __future__ import annotations

import json
import numpy as np
from typing import List, Tuple, Optional


class NumpyMLP:
    """Variable-depth MLP with tanh or relu activations.

    Parameters
    ----------
    layer_dims : list of int
        Dimensions of each layer, e.g. [obs_dim, 64, 64, act_dim].
    activation : str
        "tanh" or "relu".
    seed : int
        Random seed for weight initialisation.
    """

    def __init__(
        self,
        layer_dims: List[int],
        activation: str = "tanh",
        seed: int = 0,
    ):
        assert len(layer_dims) >= 2, "Need at least input and output dims"
        self.layer_dims = layer_dims
        self.activation = activation
        self.rng = np.random.RandomState(seed)

        # Xavier/He init
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            if activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(
                self.rng.randn(fan_in, fan_out).astype(np.float32) * scale
            )
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "relu":
            return np.maximum(0, x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def _activate_deriv(self, activated: np.ndarray) -> np.ndarray:
        """Derivative of activation given already-activated values."""
        if self.activation == "tanh":
            return 1.0 - activated ** 2
        elif self.activation == "relu":
            return (activated > 0).astype(np.float32)
        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass. Returns (output, list_of_hidden_activations).

        Note: The last layer has NO activation (logits).
        """
        hiddens = []
        h = x
        for i in range(self.n_layers - 1):
            h = self._activate(h @ self.weights[i] + self.biases[i])
            hiddens.append(h)
        # Output layer (no activation)
        out = h @ self.weights[-1] + self.biases[-1]
        return out, hiddens

    def backward(
        self,
        x: np.ndarray,
        hiddens: List[np.ndarray],
        d_output: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward pass. Returns (grad_weights, grad_biases).

        Parameters
        ----------
        x
            Input batch, shape (B, input_dim).
        hiddens
            Intermediate activations from forward().
        d_output
            Gradient w.r.t. output, shape (B, output_dim).
        """
        dw_list = []
        db_list = []

        d = d_output  # gradient flowing backwards
        # Last layer (no activation)
        prev = hiddens[-1] if hiddens else x
        dw_list.append(prev.T @ d)
        db_list.append(d.sum(axis=0))

        # Hidden layers (in reverse)
        for i in range(self.n_layers - 2, -1, -1):
            d = (d @ self.weights[i + 1].T) * self._activate_deriv(hiddens[i])
            prev = hiddens[i - 1] if i > 0 else x
            dw_list.append(prev.T @ d)
            db_list.append(d.sum(axis=0))

        # Reverse to match weight order
        dw_list.reverse()
        db_list.reverse()
        return dw_list, db_list

    def apply_gradients(
        self,
        dw_list: List[np.ndarray],
        db_list: List[np.ndarray],
        lr: float,
        ascend: bool = False,
    ):
        """Apply gradient update (SGD).

        Parameters
        ----------
        ascend : bool
            If True, perform gradient ascent (for policy optimisation).
        """
        sign = 1.0 if ascend else -1.0
        for i in range(self.n_layers):
            self.weights[i] += sign * lr * dw_list[i]
            self.biases[i] += sign * lr * db_list[i]

    # ---- Persistence ----

    def state_dict(self) -> dict:
        return {
            "layer_dims": self.layer_dims,
            "activation": self.activation,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    def load_state_dict(self, state: dict):
        self.weights = [np.array(w, dtype=np.float32) for w in state["weights"]]
        self.biases = [np.array(b, dtype=np.float32) for b in state["biases"]]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.state_dict(), f)

    @classmethod
    def load(cls, path: str) -> "NumpyMLP":
        with open(path) as f:
            state = json.load(f)
        mlp = cls(state["layer_dims"], state["activation"])
        mlp.load_state_dict(state)
        return mlp
