"""A tiny fully-connected network with hand-written forward / backward passes.

Why hand-written: the whole system must run on a bare NumPy install, and every
gradient must be verifiable by central differences (see ``tests/test_invariants``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import ModelError


def _silu(z: np.ndarray) -> np.ndarray:
    return z / (1.0 + np.exp(-z))


def _silu_grad(z: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-z))
    return s * (1.0 + z * (1.0 - s))


def fourier_embedding(c_noise: np.ndarray, dim: int = 16, max_freq: float = 4.0) -> np.ndarray:
    """Fixed (non-learned) sinusoidal time embedding.

    ``c_noise`` has shape ``(B,)``; the returned array has shape ``(B, dim)``
    with ``dim`` even.  Frequencies are log-spaced in ``[1, max_freq]``.
    """
    if dim % 2 != 0:
        raise ModelError("fourier dim must be even", dim=dim)
    half = dim // 2
    freqs = np.exp(np.linspace(0.0, float(np.log(max_freq)), half))
    arg = np.asarray(c_noise, dtype=np.float64).reshape(-1, 1) * freqs[None, :]  # (B, half)
    return np.concatenate([np.sin(arg), np.cos(arg)], axis=1)


class MLP:
    """``n_layers`` affine layers with SiLU activations on all but the last."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 96,
        n_layers: int = 3,
        fourier_dim: int = 16,
        fourier_max_freq: float = 4.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_layers < 2:
            raise ModelError("n_layers must be >= 2", n_layers=n_layers)
        rng = np.random.default_rng(0) if rng is None else rng
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.fourier_dim = int(fourier_dim)
        self.fourier_max_freq = float(fourier_max_freq)

        dims = [self.in_dim + self.fourier_dim]
        dims += [self.hidden] * (self.n_layers - 1)
        dims += [self.out_dim]
        self.dims = dims

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(self.n_layers):
            fan_in, fan_out = dims[i], dims[i + 1]
            # Output layer starts at zero: the EDM preconditioner then behaves
            # like the Gaussian-optimal denoiser D(x;sigma) = c_skip * x.
            if i == self.n_layers - 1:
                w = np.zeros((fan_in, fan_out), dtype=np.float64)
            else:
                scale = np.sqrt(2.0 / max(fan_in, 1))
                w = rng.standard_normal((fan_in, fan_out)) * scale
            self.weights.append(w)
            self.biases.append(np.zeros(fan_out, dtype=np.float64))

        # Adam state
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    # ---------------------------------------------------------------- params
    def param_vector(self) -> np.ndarray:
        return np.concatenate([w.ravel() for w in self.weights] + [b.ravel() for b in self.biases])

    def set_param_vector(self, vec: np.ndarray) -> None:
        pos = 0
        for w in self.weights:
            n = w.size
            w[...] = vec[pos : pos + n].reshape(w.shape)
            pos += n
        for b in self.biases:
            n = b.size
            b[...] = vec[pos : pos + n].reshape(b.shape)
            pos += n

    # --------------------------------------------------------------- forward
    def forward(self, x: np.ndarray, c_noise: np.ndarray) -> np.ndarray:
        """``x``: ``(B, in_dim)``; ``c_noise``: ``(B,)``.  Returns ``(B, out_dim)``."""
        emb = fourier_embedding(c_noise, self.fourier_dim, self.fourier_max_freq)
        h = np.concatenate([x, emb], axis=1)
        self._acts: list[np.ndarray] = [h]
        self._preacts: list[np.ndarray] = []
        for i in range(self.n_layers):
            z = h @ self.weights[i] + self.biases[i]
            self._preacts.append(z)
            h = _silu(z) if i < self.n_layers - 1 else z
            self._acts.append(h)
        return h

    # -------------------------------------------------------------- backward
    def backward(self, grad_out: np.ndarray) -> None:
        """Accumulate gradients in ``self.g_w`` / ``self.g_b`` (no update)."""
        g = np.asarray(grad_out, dtype=np.float64)
        self.g_w = [np.zeros_like(w) for w in self.weights]
        self.g_b = [np.zeros_like(b) for b in self.biases]
        for i in range(self.n_layers - 1, -1, -1):
            a_in = self._acts[i]
            self.g_w[i] = a_in.T @ g
            self.g_b[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.weights[i].T) * _silu_grad(self._preacts[i - 1])

    def adam_step(
        self,
        lr: float,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        grad_clip: float = 10.0,
    ) -> float:
        """One Adam update; returns the pre-clipping global gradient norm."""
        self.t += 1
        flat = np.concatenate([g.ravel() for g in self.g_w] + [g.ravel() for g in self.g_b])
        norm = float(np.sqrt(np.dot(flat, flat)))
        scale = 1.0
        if grad_clip > 0 and norm > grad_clip:
            scale = grad_clip / norm
        bc1 = 1.0 - b1**self.t
        bc2 = 1.0 - b2**self.t
        for i in range(self.n_layers):
            gw = self.g_w[i] * scale
            gb = self.g_b[i] * scale
            self.m_w[i] = b1 * self.m_w[i] + (1.0 - b1) * gw
            self.v_w[i] = b2 * self.v_w[i] + (1.0 - b2) * (gw * gw)
            self.weights[i] -= lr * (self.m_w[i] / bc1) / (np.sqrt(self.v_w[i] / bc2) + eps)
            self.m_b[i] = b1 * self.m_b[i] + (1.0 - b1) * gb
            self.v_b[i] = b2 * self.v_b[i] + (1.0 - b2) * (gb * gb)
            self.biases[i] -= lr * (self.m_b[i] / bc1) / (np.sqrt(self.v_b[i] / bc2) + eps)
        return norm

    def state_dict(self) -> dict[str, Any]:
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
            "dims": list(self.dims),
        }


__all__ = ["MLP", "fourier_embedding"]
