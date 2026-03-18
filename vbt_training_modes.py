#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vbt_training_modes.py — Training Goal Mode Policies, Set Lifecycle Manager,
Quality Gate, Next-Set Load Recommendation, and Session Report.

Design principles:
- Pure Python / numpy only (no ML deps)
- No DB writes (caller handles persistence)
- All functions are stateless and testable in isolation
- SetLifecycleManager is the only stateful class
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("vbt_training_modes")


# ══════════════════════════════════════════════════════════════
# A1 — Training Goal Mode Policies
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModePolicy:
    """Immutable policy definition for a training goal mode."""
    name: str                        # "Power" | "Strength" | "Hypertrophy"
    velocity_loss_stop_pct: float    # Stop set when velocity loss >= this
    velocity_loss_warn_pct: float    # Warn when velocity loss >= this
    velocity_range_lo: float         # m/s — lower bound of optimal zone
    velocity_range_hi: float         # m/s — upper bound of optimal zone
    rep_range_lo: int                # recommended rep range lower bound
    rep_range_hi: int                # recommended rep range upper bound
    rest_interval_s: int             # suggested rest between sets (seconds)
    load_step_kg: float              # kg increment/decrement for load progression
    description: str = ""


MODE_POLICIES: dict[str, ModePolicy] = {
    "Power": ModePolicy(
        name="Power",
        velocity_loss_stop_pct=15.0,
        velocity_loss_warn_pct=10.0,
        velocity_range_lo=0.75,
        velocity_range_hi=1.30,
        rep_range_lo=1,
        rep_range_hi=5,
        rest_interval_s=180,
        load_step_kg=2.5,
        description="爆发力训练：高速低量，速度下降 15% 即停组",
    ),
    "Strength": ModePolicy(
        name="Strength",
        velocity_loss_stop_pct=20.0,
        velocity_loss_warn_pct=12.0,
        velocity_range_lo=0.15,
        velocity_range_hi=0.50,
        rep_range_lo=2,
        rep_range_hi=6,
        rest_interval_s=240,
        load_step_kg=2.5,
        description="增力训练：中等速度，速度下降 20% 即停组",
    ),
    "Hypertrophy": ModePolicy(
        name="Hypertrophy",
        velocity_loss_stop_pct=35.0,
        velocity_loss_warn_pct=25.0,
        velocity_range_lo=0.50,
        velocity_range_hi=0.75,
        rep_range_lo=6,
        rep_range_hi=12,
        rest_interval_s=120,
        load_step_kg=2.5,
        description="肌肥大训练：高代谢压力，速度下降 35% 即停组",
    ),
}


def get_mode_policy(mode_name: str) -> ModePolicy:
    """Return mode policy; falls back to Strength if unknown."""
    policy = MODE_POLICIES.get(mode_name)
    if policy is None:
        logger.warning("未知训练模式 '%s'，使用 Strength 作为默认", mode_name)
        return MODE_POLICIES["Strength"]
    return policy


# ══════════════════════════════════════════════════════════════
# A2 — Quality Gate (Data Trust Layer)
# ══════════════════════════════════════════════════════════════

# Reason codes returned by the quality gate
QG_OK = "ok"
QG_CALIB_FALLBACK = "calibration_fallback"       # calibration used fallback/timeout method
QG_LOW_VISIBILITY = "low_visibility"              # keypoint confidence too low
QG_UNSTABLE_TRACKING = "unstable_tracking"        # high bar-path shift
QG_ROM_INCOMPLETE = "rom_incomplete"              # ROM completion below threshold
QG_VELOCITY_TOO_LOW = "velocity_too_low"          # velocity below MVT floor
QG_TIMING_UNRELIABLE = "timing_unreliable"        # timing source flagged unreliable

QUALITY_THRESHOLDS = {
    "min_rom_completion_pct": 60.0,
    "min_velocity_m_s": 0.05,
    "max_bar_shift_cm": 25.0,
    "trusted_calib_methods": {"shoulder_ankle", "head_ankle"},
}


@dataclass
class QualityResult:
    """Result of quality gate evaluation for a single rep."""
    trusted: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def reason_str(self) -> str:
        return ",".join(self.reasons) if self.reasons else QG_OK


def evaluate_rep_quality(
    v_mean: float,
    rom_completion_pct: Optional[float],
    bar_shift_cm: Optional[float],
    calib_method: Optional[str],
    calib_is_fallback: Optional[bool],
    timing_source: Optional[str] = None,
    pose_issues: Optional[str] = None,
) -> QualityResult:
    """
    Evaluate whether a rep's data is trustworthy enough for analytics/modeling.
    Returns QualityResult(trusted=True/False, reasons=[...]).
    """
    reasons: list[str] = []

    # Calibration trust check
    if calib_is_fallback:
        reasons.append(QG_CALIB_FALLBACK)
    elif calib_method is not None and calib_method not in QUALITY_THRESHOLDS["trusted_calib_methods"]:
        reasons.append(QG_CALIB_FALLBACK)

    # Velocity floor
    if v_mean is not None and v_mean < QUALITY_THRESHOLDS["min_velocity_m_s"]:
        reasons.append(QG_VELOCITY_TOO_LOW)

    # ROM completeness
    if rom_completion_pct is not None and rom_completion_pct < QUALITY_THRESHOLDS["min_rom_completion_pct"]:
        reasons.append(QG_ROM_INCOMPLETE)

    # Bar path stability
    if bar_shift_cm is not None and bar_shift_cm > QUALITY_THRESHOLDS["max_bar_shift_cm"]:
        reasons.append(QG_UNSTABLE_TRACKING)

    # Pose visibility (derived from pose_issues field)
    if pose_issues:
        if "unstable" in pose_issues:
            reasons.append(QG_LOW_VISIBILITY)

    trusted = len(reasons) == 0
    return QualityResult(trusted=trusted, reasons=reasons)


# ══════════════════════════════════════════════════════════════
# A3 — Set Lifecycle Manager
# ══════════════════════════════════════════════════════════════

SET_START_MIN_REPS = 1          # at least 1 rep to consider set started
SET_END_IDLE_S = 20.0           # seconds of inactivity to auto-end set
REST_READY_FACTOR = 0.9         # ready for next set when rest >= factor * suggested


@dataclass
class SetSummary:
    """Summary of a completed set."""
    set_number: int
    session_id: str
    user_name: str
    load_kg: float
    mode: str
    reps: int
    rep_velocities: list[float]
    best_velocity: float
    mean_velocity: float
    velocity_loss_pct: float
    trusted_rep_count: int
    untrusted_rep_count: int
    quality_reasons: list[str]   # unique quality issues across reps
    start_ts: float              # time.monotonic()
    end_ts: float
    duration_s: float
    was_stopped_by_policy: bool


class SetLifecycleManager:
    """
    Tracks the state of the current training set.
    States: idle → active → resting → ready
    """
    STATES = ("idle", "active", "resting", "ready")

    def __init__(self, mode: str = "Strength", rest_override_s: Optional[int] = None) -> None:
        self.mode = mode
        self.policy: ModePolicy = get_mode_policy(mode)
        self.rest_override_s = rest_override_s
        self._state: str = "idle"
        self._set_number: int = 0
        self._session_id: str = ""
        self._user_name: str = "qiqi"
        self._load_kg: float = 0.0
        self._rep_velocities: list[float] = []
        self._rep_quality_results: list[QualityResult] = []
        self._set_start_ts: float = 0.0
        self._set_end_ts: float = 0.0
        self._rest_start_ts: float = 0.0
        self._last_rep_ts: float = 0.0
        self._set_history: list[SetSummary] = []
        self._stopped_by_policy: bool = False

    # ── public API ────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def set_number(self) -> int:
        return self._set_number

    @property
    def rest_elapsed_s(self) -> float:
        if self._state not in ("resting", "ready"):
            return 0.0
        return time.monotonic() - self._rest_start_ts

    @property
    def rest_remaining_s(self) -> float:
        target = self.rest_override_s or self.policy.rest_interval_s
        remaining = target - self.rest_elapsed_s
        return max(0.0, remaining)

    @property
    def suggested_rest_s(self) -> int:
        return self.rest_override_s or self.policy.rest_interval_s

    @property
    def set_history(self) -> list[SetSummary]:
        return list(self._set_history)

    def start_session(self, session_id: str, user_name: str, load_kg: float) -> None:
        """Call once at the beginning of a training session."""
        self._session_id = session_id
        self._user_name = user_name
        self._load_kg = load_kg
        self._state = "idle"
        self._set_number = 0
        self._set_history.clear()
        logger.info("Set lifecycle session started: session=%s user=%s", session_id, user_name)

    def on_rep_completed(
        self,
        v_mean: float,
        quality: QualityResult,
        load_kg: Optional[float] = None,
    ) -> None:
        """Call after each rep is completed."""
        ts = time.monotonic()
        if self._state == "idle":
            self._set_number += 1
            self._set_start_ts = ts
            self._rep_velocities.clear()
            self._rep_quality_results.clear()
            self._stopped_by_policy = False
            self._state = "active"
            logger.info("Set %d started", self._set_number)
        if load_kg is not None:
            self._load_kg = load_kg
        self._rep_velocities.append(float(v_mean))
        self._rep_quality_results.append(quality)
        self._last_rep_ts = ts

        # Check velocity-loss stop condition
        if len(self._rep_velocities) >= 2:
            best = max(self._rep_velocities)
            curr = self._rep_velocities[-1]
            if best > 0:
                loss_pct = (best - curr) / best * 100.0
                if loss_pct >= self.policy.velocity_loss_stop_pct:
                    logger.info(
                        "Set %d: policy stop triggered (loss=%.1f%% >= %.1f%%)",
                        self._set_number, loss_pct, self.policy.velocity_loss_stop_pct,
                    )
                    self._stopped_by_policy = True

    def on_set_ended(self) -> Optional[SetSummary]:
        """Explicitly end the current set. Returns SetSummary or None if no reps."""
        if self._state != "active" or not self._rep_velocities:
            return None
        summary = self._build_summary()
        self._set_history.append(summary)
        self._state = "resting"
        self._rest_start_ts = time.monotonic()
        self._set_end_ts = time.monotonic()
        logger.info("Set %d ended: reps=%d best_vel=%.3f", summary.set_number, summary.reps, summary.best_velocity)
        return summary

    def tick(self) -> Optional[SetSummary]:
        """
        Call periodically (e.g., every second) to handle auto-end and rest transitions.
        Returns SetSummary if a set was auto-ended due to idle timeout, else None.
        """
        now = time.monotonic()
        auto_ended: Optional[SetSummary] = None

        if self._state == "active" and self._last_rep_ts > 0:
            if (now - self._last_rep_ts) >= SET_END_IDLE_S:
                logger.info("Set %d: auto-ended due to %.0fs inactivity", self._set_number, SET_END_IDLE_S)
                auto_ended = self.on_set_ended()

        if self._state == "resting":
            elapsed = now - self._rest_start_ts
            target = self.rest_override_s or self.policy.rest_interval_s
            if elapsed >= target * REST_READY_FACTOR:
                self._state = "ready"
                logger.info("Set %d rest complete (%.0fs elapsed)", self._set_number, elapsed)

        return auto_ended

    def reset_for_next_set(self, load_kg: Optional[float] = None) -> None:
        """Call when user is ready to start the next set."""
        if load_kg is not None:
            self._load_kg = load_kg
        self._rep_velocities.clear()
        self._rep_quality_results.clear()
        self._stopped_by_policy = False
        self._state = "idle"

    # ── private ───────────────────────────────────────────────

    def _build_summary(self) -> SetSummary:
        vels = self._rep_velocities
        best = float(max(vels)) if vels else 0.0
        mean = float(np.mean(vels)) if vels else 0.0
        last = float(vels[-1]) if vels else 0.0
        loss_pct = (best - last) / best * 100.0 if best > 0 else 0.0
        trusted = sum(1 for q in self._rep_quality_results if q.trusted)
        untrusted = len(self._rep_quality_results) - trusted
        all_reasons: list[str] = []
        for q in self._rep_quality_results:
            all_reasons.extend(q.reasons)
        now = time.monotonic()
        return SetSummary(
            set_number=self._set_number,
            session_id=self._session_id,
            user_name=self._user_name,
            load_kg=self._load_kg,
            mode=self.mode,
            reps=len(vels),
            rep_velocities=list(vels),
            best_velocity=best,
            mean_velocity=mean,
            velocity_loss_pct=loss_pct,
            trusted_rep_count=trusted,
            untrusted_rep_count=untrusted,
            quality_reasons=list(dict.fromkeys(all_reasons)),
            start_ts=self._set_start_ts,
            end_ts=now,
            duration_s=now - self._set_start_ts,
            was_stopped_by_policy=self._stopped_by_policy,
        )


# ══════════════════════════════════════════════════════════════
# A4 — Next-Set Load Recommendation
# ══════════════════════════════════════════════════════════════

@dataclass
class LoadRecommendation:
    action: str            # "increase" | "maintain" | "decrease" | "stop"
    delta_kg: float        # positive = increase, negative = decrease, 0 = maintain
    new_load_kg: float
    reason: str
    confidence: float      # 0.0–1.0
    safety_override: bool  # True if safety rules forced the decision
    contributing_factors: list[str] = field(default_factory=list)


def recommend_next_load(
    summary: SetSummary,
    policy: ModePolicy,
    fatigue_risk: float = 0.0,
    stop_probability: float = 0.0,
    technique_anomaly: float = 0.0,
    dl_confidence: float = 0.0,
) -> LoadRecommendation:
    """
    Hybrid decision: safety rules first, then DL-weighted policy.

    Safety rules (hard constraints, always override DL):
    1. If stop_probability >= 0.85 or fatigue_risk >= 0.90 -> stop
    2. If velocity_loss >= stop threshold -> stop
    3. If best_velocity < 0.10 m/s (calibration error floor) -> maintain

    DL-weighted policy:
    - Uses fatigue_risk and stop_probability to bias toward decrease/stop
    - Falls back to deterministic rules when dl_confidence < 0.3
    """
    load = summary.load_kg
    best_vel = summary.best_velocity
    loss_pct = summary.velocity_loss_pct
    reps = summary.reps
    factors: list[str] = []

    # ── SAFETY FLOOR (hard constraints) ──────────────────────
    if best_vel < 0.10:
        return LoadRecommendation(
            action="maintain", delta_kg=0.0, new_load_kg=load,
            reason="最大速度异常低，可能标定错误，保持当前负重",
            confidence=0.9, safety_override=True,
            contributing_factors=["velocity_floor_breach"],
        )

    if stop_probability >= 0.85 and dl_confidence >= 0.3:
        return LoadRecommendation(
            action="stop", delta_kg=0.0, new_load_kg=load,
            reason=f"AI 停组概率 {stop_probability:.0%} 超过安全阈值，建议结束训练",
            confidence=dl_confidence, safety_override=True,
            contributing_factors=["dl_stop_probability"],
        )

    if fatigue_risk >= 0.90 and dl_confidence >= 0.3:
        return LoadRecommendation(
            action="stop", delta_kg=0.0, new_load_kg=load,
            reason=f"AI 疲劳风险 {fatigue_risk:.0%} 极高，强烈建议停止",
            confidence=dl_confidence, safety_override=True,
            contributing_factors=["dl_fatigue_risk"],
        )

    if loss_pct >= policy.velocity_loss_stop_pct:
        return LoadRecommendation(
            action="stop", delta_kg=0.0, new_load_kg=load,
            reason=f"速度流失 {loss_pct:.1f}% 超过 {policy.name} 模式阈值 {policy.velocity_loss_stop_pct:.0f}%",
            confidence=0.95, safety_override=True,
            contributing_factors=["velocity_loss_threshold"],
        )

    # ── AI-weighted decision ──────────────────────────────────
    # Aggregate fatigue signal (DL if available, else pure rule)
    use_dl = dl_confidence >= 0.3
    effective_fatigue = fatigue_risk if use_dl else (loss_pct / 100.0)
    effective_stop = stop_probability if use_dl else 0.0

    if use_dl:
        factors.append(f"dl_fatigue={fatigue_risk:.2f}")
        factors.append(f"dl_stop_prob={stop_probability:.2f}")
        if technique_anomaly > 0.5:
            factors.append(f"technique_anomaly={technique_anomaly:.2f}")
    else:
        factors.append("rule_based_fallback")
        factors.append(f"velocity_loss={loss_pct:.1f}%")

    # Decrease path
    if effective_fatigue >= 0.7 or (loss_pct >= policy.velocity_loss_warn_pct and reps < policy.rep_range_lo):
        delta = -policy.load_step_kg
        new_load = max(0.0, load + delta)
        reason_parts = []
        if effective_fatigue >= 0.7:
            reason_parts.append(f"疲劳信号高 ({effective_fatigue:.0%})")
        if loss_pct >= policy.velocity_loss_warn_pct:
            reason_parts.append(f"速度流失 {loss_pct:.1f}%")
        return LoadRecommendation(
            action="decrease", delta_kg=delta, new_load_kg=new_load,
            reason="；".join(reason_parts) + f"，建议减重 {abs(delta):.1f} kg",
            confidence=max(effective_fatigue, 0.5), safety_override=False,
            contributing_factors=factors,
        )

    # Increase path
    if (
        best_vel > policy.velocity_range_hi
        and reps >= policy.rep_range_hi
        and loss_pct < policy.velocity_loss_warn_pct * 0.5
        and effective_fatigue < 0.4
    ):
        delta = policy.load_step_kg
        new_load = load + delta
        factors.append(f"velocity_in_upper_zone={best_vel:.3f}")
        return LoadRecommendation(
            action="increase", delta_kg=delta, new_load_kg=new_load,
            reason=f"速度 {best_vel:.3f} m/s 高于区间上限，完成推荐 rep 数，状态良好，建议加重 {delta:.1f} kg",
            confidence=0.75, safety_override=False,
            contributing_factors=factors,
        )

    # Maintain
    return LoadRecommendation(
        action="maintain", delta_kg=0.0, new_load_kg=load,
        reason=f"当前状态良好（速度流失 {loss_pct:.1f}%），保持 {load:.1f} kg",
        confidence=0.7, safety_override=False,
        contributing_factors=factors,
    )


# ══════════════════════════════════════════════════════════════
# A5 — Session Report
# ══════════════════════════════════════════════════════════════

@dataclass
class SessionReport:
    """Actionable end-of-session summary."""
    session_id: str
    user_name: str
    mode: str
    total_sets: int
    total_reps: int
    total_trusted_reps: int
    best_mean_velocity: float
    velocity_trend: list[float]         # best velocity per set
    velocity_loss_trend: list[float]    # velocity_loss_pct per set
    technique_consistency_trend: list[float]  # trusted_rep% per set
    fatigue_risk_score: float           # 0–1, derived from velocity decay
    quality_rejection_rate: float       # fraction of untrusted reps
    next_session_recommendation: str
    set_summaries: list[SetSummary]


def build_session_report(
    session_id: str,
    user_name: str,
    mode: str,
    set_summaries: list[SetSummary],
) -> SessionReport:
    """
    Build an actionable session report from completed set summaries.
    No DB access — caller passes data in.
    """
    if not set_summaries:
        return SessionReport(
            session_id=session_id, user_name=user_name, mode=mode,
            total_sets=0, total_reps=0, total_trusted_reps=0,
            best_mean_velocity=0.0, velocity_trend=[], velocity_loss_trend=[],
            technique_consistency_trend=[], fatigue_risk_score=0.0,
            quality_rejection_rate=0.0,
            next_session_recommendation="数据不足，请至少完成一组训练后生成报告。",
            set_summaries=[],
        )

    total_reps = sum(s.reps for s in set_summaries)
    total_trusted = sum(s.trusted_rep_count for s in set_summaries)
    best_vel_per_set = [s.best_velocity for s in set_summaries]
    loss_per_set = [s.velocity_loss_pct for s in set_summaries]
    consistency_per_set = [
        s.trusted_rep_count / s.reps if s.reps > 0 else 0.0
        for s in set_summaries
    ]
    best_mean_vel = max(best_vel_per_set) if best_vel_per_set else 0.0
    rejection_rate = (total_reps - total_trusted) / total_reps if total_reps > 0 else 0.0

    # Fatigue risk: linear decay of best velocity across sets
    fatigue_score = 0.0
    if len(best_vel_per_set) >= 2:
        first_vel = best_vel_per_set[0]
        last_vel = best_vel_per_set[-1]
        if first_vel > 0:
            decay = (first_vel - last_vel) / first_vel
            fatigue_score = float(np.clip(decay, 0.0, 1.0))

    # Next session recommendation
    policy = get_mode_policy(mode)
    if fatigue_score >= 0.30:
        rec = f"疲劳风险较高（跨组速度衰减 {fatigue_score:.0%}），下次训练建议减重 {policy.load_step_kg:.1f} kg 或增加休息日。"
    elif fatigue_score < 0.10 and best_mean_vel > policy.velocity_range_hi * 0.9:
        rec = f"状态极佳，速度稳定（最佳均速 {best_mean_vel:.3f} m/s），下次训练可尝试增加 {policy.load_step_kg:.1f} kg。"
    else:
        rec = f"训练状态正常（最佳均速 {best_mean_vel:.3f} m/s，跨组衰减 {fatigue_score:.0%}），保持当前负重继续巩固。"

    return SessionReport(
        session_id=session_id,
        user_name=user_name,
        mode=mode,
        total_sets=len(set_summaries),
        total_reps=total_reps,
        total_trusted_reps=total_trusted,
        best_mean_velocity=best_mean_vel,
        velocity_trend=best_vel_per_set,
        velocity_loss_trend=loss_per_set,
        technique_consistency_trend=consistency_per_set,
        fatigue_risk_score=round(fatigue_score, 3),
        quality_rejection_rate=round(rejection_rate, 3),
        next_session_recommendation=rec,
        set_summaries=set_summaries,
    )
 