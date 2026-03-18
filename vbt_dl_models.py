#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vbt_dl_models.py — Lightweight Deep Learning Modules for Edge Deployment.

Modules:
  A) SetFatigueNet   — 1D temporal model: predict fatigue_risk + stop_probability per set
  B) TechniqueAnomalyNet — compact autoencoder: detect technique drift
  C) InferenceWrapper — unified interface with fallback to deterministic rules

Inference: pure numpy forward pass of exported .npz weights.
Fallback: deterministic rules when model file is missing/corrupt.
Export: weights saved as .npz; optional TFLite export via separate training script.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("vbt_dl_models")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FATIGUE_MODEL_FILE = "set_fatigue_net.npz"
TECHNIQUE_MODEL_FILE = "technique_anomaly_net.npz"
FATIGUE_INPUT_DIM = 8
TECHNIQUE_INPUT_DIM = 16
ANOMALY_WARN_THRESHOLD = 0.35
ANOMALY_HIGH_THRESHOLD = 0.65


@dataclass
class FatiguePrediction:
    fatigue_risk: float
    stop_probability: float
    confidence: float
    used_model: bool
    model_version: str


@dataclass
class TechniquePrediction:
    technique_anomaly_score: float
    severity: str
    used_model: bool
    model_version: str
    reconstruction_error: float


# ── SetFatigueNet ─────────────────────────────────────────────

class SetFatigueNet:
    """
    1D Conv temporal model for set-level fatigue prediction.
    Architecture: Conv1D(k=3,16) -> GAP -> Dense(8) -> Dense(2,sigmoid)
    All weights stored in .npz; forward pass is pure numpy.
    """
    VERSION = "v1.0"

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._loaded = False
        self._weights: dict = {}
        self._load_error: Optional[str] = None
        path = model_path or os.path.join(MODELS_DIR, FATIGUE_MODEL_FILE)
        self._path = path
        self._load(path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            self._load_error = f"file not found: {path}"
            logger.info("SetFatigueNet: model not found at %s, using fallback", path)
            return
        try:
            data = np.load(path, allow_pickle=False)
            required = {"conv_w", "conv_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b"}
            if not required.issubset(set(data.files)):
                missing = required - set(data.files)
                self._load_error = f"missing weight keys: {missing}"
                logger.warning("SetFatigueNet: missing keys %s", missing)
                return
            self._weights = {k: data[k] for k in data.files}
            self._loaded = True
            self._load_error = None
            logger.info("SetFatigueNet: loaded from %s", path)
        except Exception as e:
            self._load_error = str(e)
            logger.warning("SetFatigueNet: load failed: %s", e)

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def is_available(self) -> bool:
        return self._loaded

    def _forward(self, x: np.ndarray) -> np.ndarray:
        w = self._weights
        conv_w = w["conv_w"]  # (K, 8, 16)
        conv_b = w["conv_b"]  # (16,)
        K = conv_w.shape[0]
        T = x.shape[0]
        if T < K:
            x = np.pad(x, ((K - T, 0), (0, 0)), mode="edge")
            T = x.shape[0]
        out_len = T - K + 1
        conv_out = np.zeros((out_len, 16), dtype=np.float32)
        for t in range(out_len):
            conv_out[t] = np.einsum("ki,kio->o", x[t:t + K], conv_w) + conv_b
        conv_out = np.maximum(0.0, conv_out)
        pooled = conv_out.mean(axis=0)
        h1 = np.maximum(0.0, pooled @ w["fc1_w"] + w["fc1_b"])
        logits = h1 @ w["fc2_w"] + w["fc2_b"]
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, rep_features: list[list[float]]) -> FatiguePrediction:
        if not rep_features:
            return FatiguePrediction(0.0, 0.0, 0.0, False, "rule_fallback")
        x = np.array(rep_features, dtype=np.float32)
        x[:, 0] = np.clip(x[:, 0] / 1.5, 0, 1)
        x[:, 1] = np.clip(x[:, 1] / 2.0, 0, 1)
        x[:, 2] = np.clip(x[:, 2] / 0.6, 0, 1)
        x[:, 3] = np.clip(x[:, 3] / 50.0, 0, 1)
        x[:, 4] = np.clip(x[:, 4] / 3.0, 0, 1)
        x[:, 5] = np.clip(x[:, 5] / 20.0, 0, 1)
        if self._loaded:
            try:
                out = self._forward(x)
                n = len(rep_features)
                conf = float(np.clip(0.4 + 0.06 * n, 0.0, 0.95))
                return FatiguePrediction(
                    float(np.clip(out[0], 0, 1)),
                    float(np.clip(out[1], 0, 1)),
                    conf, True, self.VERSION,
                )
            except Exception as e:
                logger.warning("SetFatigueNet forward failed: %s", e)
        return _fatigue_rule_fallback(rep_features)

    @staticmethod
    def build_feature_vector(
        v_mean: float, v_peak: Optional[float], rom: float,
        velocity_loss_pct: float, rep_duration_s: float,
        bar_shift_cm: Optional[float], pose_issue_score: float,
        calib_is_fallback: bool,
    ) -> list[float]:
        return [
            float(v_mean),
            float(v_peak) if v_peak is not None else float(v_mean),
            float(rom),
            float(velocity_loss_pct),
            float(rep_duration_s),
            float(bar_shift_cm) if bar_shift_cm is not None else 0.0,
            float(pose_issue_score),
            0.0 if calib_is_fallback else 1.0,
        ]


def _fatigue_rule_fallback(rep_features: list[list[float]]) -> FatiguePrediction:
    vels = [f[0] for f in rep_features]
    loss_pcts = [f[3] for f in rep_features]
    best_v = max(vels) if vels else 0.0
    last_v = vels[-1] if vels else 0.0
    last_loss = loss_pcts[-1] if loss_pcts else 0.0
    fatigue = float(np.clip((best_v - last_v) / best_v if best_v > 0 else 0.0, 0.0, 1.0))
    stop_p = float(np.clip(last_loss / 50.0, 0.0, 1.0))
    return FatiguePrediction(fatigue, stop_p, 0.2, False, "rule_fallback")


# ── TechniqueAnomalyNet ───────────────────────────────────────

class TechniqueAnomalyNet:
    """
    Autoencoder for technique anomaly detection.
    Architecture: Enc(16->8->4) Dec(4->8->16)
    Anomaly score = normalised reconstruction MSE.
    """
    VERSION = "v1.0"

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._loaded = False
        self._weights: dict = {}
        self._baseline_error: float = 1.0
        self._load_error: Optional[str] = None
        path = model_path or os.path.join(MODELS_DIR, TECHNIQUE_MODEL_FILE)
        self._path = path
        self._load(path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            self._load_error = f"file not found: {path}"
            logger.info("TechniqueAnomalyNet: not found at %s, using fallback", path)
            return
        try:
            data = np.load(path, allow_pickle=False)
            required = {"enc1_w", "enc1_b", "enc2_w", "enc2_b",
                        "dec1_w", "dec1_b", "dec2_w", "dec2_b", "baseline_error"}
            if not required.issubset(set(data.files)):
                missing = required - set(data.files)
                self._load_error = f"missing weight keys: {missing}"
                logger.warning("TechniqueAnomalyNet: missing keys %s", missing)
                return
            self._weights = {k: data[k] for k in data.files if k != "baseline_error"}
            self._baseline_error = float(data["baseline_error"])
            self._loaded = True
            self._load_error = None
            logger.info("TechniqueAnomalyNet: loaded (baseline_err=%.4f)", self._baseline_error)
        except Exception as e:
            self._load_error = str(e)
            logger.warning("TechniqueAnomalyNet: load failed: %s", e)

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def is_available(self) -> bool:
        return self._loaded

    def _forward(self, x: np.ndarray) -> np.ndarray:
        w = self._weights
        h = np.maximum(0.0, x @ w["enc1_w"] + w["enc1_b"])
        z = np.maximum(0.0, h @ w["enc2_w"] + w["enc2_b"])
        h2 = np.maximum(0.0, z @ w["dec1_w"] + w["dec1_b"])
        return h2 @ w["dec2_w"] + w["dec2_b"]

    def predict(self, trajectory_features: list[float]) -> TechniquePrediction:
        if not trajectory_features:
            return TechniquePrediction(0.0, "normal", False, "rule_fallback", 0.0)
        x = np.array(trajectory_features, dtype=np.float32)
        if x.shape[0] != TECHNIQUE_INPUT_DIM:
            x = np.pad(x, (0, max(0, TECHNIQUE_INPUT_DIM - x.shape[0])))[:TECHNIQUE_INPUT_DIM]
        x_norm = (x - x.mean()) / (x.std() + 1e-8)
        if self._loaded:
            try:
                recon = self._forward(x_norm)
                raw_err = float(np.mean((x_norm - recon) ** 2))
                score = float(np.clip(raw_err / (self._baseline_error * 3.0 + 1e-8), 0.0, 1.0))
                severity = (
                    "high-risk" if score >= ANOMALY_HIGH_THRESHOLD
                    else "warning" if score >= ANOMALY_WARN_THRESHOLD
                    else "normal"
                )
                return TechniquePrediction(score, severity, True, self.VERSION, raw_err)
            except Exception as e:
                logger.warning("TechniqueAnomalyNet forward failed: %s", e)
        return TechniquePrediction(0.0, "normal", False, "rule_fallback", 0.0)

    @staticmethod
    def build_feature_vector(
        rep_velocities: list[float],
        knee_angles: list[float],
        trunk_angles: list[float],
        rom_values: list[float],
    ) -> list[float]:
        """
        Build a 16-dim feature vector summarising one rep's kinematics.
        Pads or truncates each input signal to 4 values.
        """
        def _summarise(vals: list[float], n: int = 4) -> list[float]:
            if not vals:
                return [0.0] * n
            arr = np.array(vals, dtype=np.float32)
            idx = np.linspace(0, len(arr) - 1, n).astype(int)
            return arr[idx].tolist()

        return (
            _summarise(rep_velocities) +
            _summarise(knee_angles) +
            _summarise(trunk_angles) +
            _summarise(rom_values)
        )


# ── InferenceWrapper (unified API) ────────────────────────────

class InferenceWrapper:
    """
    Unified inference interface.
    Loads both models; provides .predict_fatigue() and .predict_technique().
    If a model file is missing, transparently falls back to deterministic rules.
    Logs fallback events with reason code for monitoring.
    """

    def __init__(
        self,
        fatigue_model_path: Optional[str] = None,
        technique_model_path: Optional[str] = None,
    ) -> None:
        self.fatigue_net = SetFatigueNet(fatigue_model_path)
        self.technique_net = TechniqueAnomalyNet(technique_model_path)
        self._fallback_count_fatigue = 0
        self._fallback_count_technique = 0

    @property
    def fatigue_model_available(self) -> bool:
        return self.fatigue_net.is_available

    @property
    def technique_model_available(self) -> bool:
        return self.technique_net.is_available

    def predict_fatigue(self, rep_features: list[list[float]]) -> FatiguePrediction:
        result = self.fatigue_net.predict(rep_features)
        if not result.used_model:
            self._fallback_count_fatigue += 1
            if self._fallback_count_fatigue == 1:
                logger.info("SetFatigueNet: using deterministic fallback (model not loaded)")
        return result

    def predict_technique(
        self,
        rep_velocities: list[float],
        knee_angles: list[float],
        trunk_angles: list[float],
        rom_values: list[float],
    ) -> TechniquePrediction:
        feat = TechniqueAnomalyNet.build_feature_vector(
            rep_velocities, knee_angles, trunk_angles, rom_values
        )
        result = self.technique_net.predict(feat)
        if not result.used_model:
            self._fallback_count_technique += 1
            if self._fallback_count_technique == 1:
                logger.info("TechniqueAnomalyNet: using deterministic fallback (model not loaded)")
        return result

    def fallback_stats(self) -> dict:
        return {
            "fatigue_fallback_count": self._fallback_count_fatigue,
            "technique_fallback_count": self._fallback_count_technique,
            "fatigue_model_available": self.fatigue_model_available,
            "technique_model_available": self.technique_model_available,
        }


# ── Module-level singleton ────────────────────────────────────
_inference: Optional[InferenceWrapper] = None


def get_inference() -> InferenceWrapper:
    """Return (and lazily initialise) the module-level inference wrapper."""
    global _inference
    if _inference is None:
        _inference = InferenceWrapper()
    return _inference
