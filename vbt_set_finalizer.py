#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vbt_set_finalizer.py — Stateless set-end pipeline.

On every set completion (from CV engine or video processor), call:
    finalize_set(session_id, set_number, user_name, mode, load_kg, rep_rows, db_path)

This function:
  1. Aggregates rep_rows into a SetSummary
  2. Runs AI inference (SetFatigueNet + TechniqueAnomalyNet) with fallback
  3. Runs load recommendation
  4. Persists to `sets` table (via persist_set_summary)
  5. Persists to `prediction_logs` table (via log_prediction)
  6. Returns a FinalizedSet dataclass for in-memory use

Design: no global state, always writes even in fallback mode.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("vbt_set_finalizer")


@dataclass
class FinalizedSet:
    set_number: int
    session_id: str
    reps: int
    best_velocity: float
    mean_velocity: float
    velocity_loss_pct: float
    fatigue_risk: float
    stop_probability: float
    technique_anomaly: float
    recommendation_action: str
    recommendation_reason: str
    fatigue_model_status: str   # "loaded" | "fallback" | "error"
    tech_model_status: str
    set_row_id: Optional[int]


def finalize_set(
    session_id: str,
    set_number: int,
    user_name: str,
    mode: str,
    load_kg: float,
    rep_rows: list[dict],
    db_path: str,
) -> Optional[FinalizedSet]:
    """
    rep_rows: list of dicts with keys:
        v_mean, rom, velocity_loss, calib_is_fallback, pose_issues,
        left_knee_angle, right_knee_angle, trunk_angle, bar_shift_cm (optional)
    """
    if not rep_rows:
        logger.warning("finalize_set: no reps for set %d, skipping", set_number)
        return None

    # ── 1. Import dependencies (lazy to avoid circular) ───────
    try:
        from vbt_training_modes import (
            SetSummary, evaluate_rep_quality, get_mode_policy, recommend_next_load,
        )
        from vbt_dl_models import SetFatigueNet, TechniqueAnomalyNet, get_inference
        from vbt_analytics_pro import persist_set_summary, log_prediction, _enable_wal
        import sqlite3, numpy as np
    except Exception as e:
        logger.error("finalize_set: import failed: %s", e)
        return None

    # ── 2. Build quality-annotated rep list ───────────────────
    vels = [float(r.get("v_mean") or 0) for r in rep_rows]
    roms = [float(r.get("rom") or 0) for r in rep_rows]
    best_v = max(vels) if vels else 0.0
    mean_v = float(sum(vels) / len(vels)) if vels else 0.0
    last_v = vels[-1] if vels else 0.0
    loss_pct = (best_v - last_v) / best_v * 100.0 if best_v > 0 else 0.0

    quality_results = []
    for r in rep_rows:
        qr = evaluate_rep_quality(
            v_mean=float(r.get("v_mean") or 0),
            rom_completion_pct=float(r.get("rom_completion_pct") or 100),
            bar_shift_cm=float(r.get("bar_shift_cm") or 0),
            calib_method=r.get("calib_method"),
            calib_is_fallback=bool(r.get("calib_is_fallback")),
            timing_source=r.get("timing_source"),
            pose_issues=r.get("pose_issues"),
        )
        quality_results.append(qr)

    trusted_count = sum(1 for q in quality_results if q.trusted)
    untrusted_count = len(quality_results) - trusted_count
    all_reasons: list[str] = []
    for q in quality_results:
        all_reasons.extend(q.reasons)
    unique_reasons = list(dict.fromkeys(all_reasons))

    now = time.monotonic()
    summary = SetSummary(
        set_number=set_number,
        session_id=session_id,
        user_name=user_name,
        load_kg=load_kg,
        mode=mode,
        reps=len(rep_rows),
        rep_velocities=vels,
        best_velocity=best_v,
        mean_velocity=mean_v,
        velocity_loss_pct=round(loss_pct, 2),
        trusted_rep_count=trusted_count,
        untrusted_rep_count=untrusted_count,
        quality_reasons=unique_reasons,
        start_ts=now - 60.0,  # approximate; real start_ts not tracked here
        end_ts=now,
        duration_s=60.0,
        was_stopped_by_policy=False,
    )

    # ── 3. AI inference (always runs, falls back gracefully) ──
    inf = get_inference()

    # Build fatigue feature vectors
    feat_vecs = []
    for r in rep_rows:
        fv = SetFatigueNet.build_feature_vector(
            v_mean=float(r.get("v_mean") or 0),
            v_peak=float(r.get("v_mean") or 0),
            rom=float(r.get("rom") or 0),
            velocity_loss_pct=float(r.get("velocity_loss") or 0),
            rep_duration_s=0.5,
            bar_shift_cm=float(r.get("bar_shift_cm") or 0),
            pose_issue_score=0.3 if r.get("pose_issues") else 0.0,
            calib_is_fallback=bool(r.get("calib_is_fallback")),
        )
        feat_vecs.append(fv)

    fat_pred = inf.predict_fatigue(feat_vecs)
    fat_status = "loaded" if fat_pred.used_model else "fallback"

    # Build technique feature vectors
    knee_angles = [float(r.get("left_knee_angle") or 90) for r in rep_rows]
    trunk_angles = [float(r.get("trunk_angle") or 15) for r in rep_rows]
    tech_pred = inf.predict_technique(vels, knee_angles, trunk_angles, roms)
    tech_status = "loaded" if tech_pred.used_model else "fallback"

    # ── 4. Load recommendation ────────────────────────────────
    policy = get_mode_policy(mode)
    rec = recommend_next_load(
        summary, policy,
        fatigue_risk=fat_pred.fatigue_risk,
        stop_probability=fat_pred.stop_probability,
        technique_anomaly=tech_pred.technique_anomaly_score,
        dl_confidence=fat_pred.confidence,
    )

    # ── 5. Persist set summary ────────────────────────────────
    # Update summary with recommendation
    set_row_id: Optional[int] = None
    try:
        conn = sqlite3.connect(db_path)
        _enable_wal(conn)
        from datetime import datetime as _dt
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO sets (
                session_id, user_name, set_number, mode, load_kg, reps,
                best_velocity, mean_velocity, velocity_loss_pct,
                trusted_rep_count, untrusted_rep_count, quality_reasons,
                started_at, ended_at, was_stopped_by_policy,
                load_recommendation, recommendation_reason
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'),?,?,?)
        """, (
            session_id, user_name, set_number, mode, load_kg, len(rep_rows),
            best_v, mean_v, round(loss_pct, 2),
            trusted_count, untrusted_count, ",".join(unique_reasons),
            0, rec.action, rec.reason,
        ))
        conn.commit()
        set_row_id = cur.lastrowid
        conn.close()
        logger.info("Set %d persisted (id=%s, action=%s)", set_number, set_row_id, rec.action)
    except Exception as e:
        logger.error("persist sets row failed: %s", e)

    # ── 6. Persist prediction log ─────────────────────────────
    try:
        log_prediction(
            db_path=db_path,
            session_id=session_id,
            set_id=set_row_id,
            model_name="SetFatigueNet+TechniqueAnomalyNet",
            model_version=f"{fat_pred.model_version}/{tech_pred.model_version}",
            prediction_type="set_finalization",
            fatigue_risk=fat_pred.fatigue_risk,
            stop_probability=fat_pred.stop_probability,
            technique_anomaly=tech_pred.technique_anomaly_score,
            confidence=fat_pred.confidence,
            used_model=fat_pred.used_model or tech_pred.used_model,
            recommendation=rec.action,
            recommendation_reason=rec.reason,
        )
        logger.info("Prediction log written for set %d (fallback=%s)",
                    set_number, fat_status)
    except Exception as e:
        logger.error("log_prediction failed: %s", e)

    return FinalizedSet(
        set_number=set_number,
        session_id=session_id,
        reps=len(rep_rows),
        best_velocity=best_v,
        mean_velocity=mean_v,
        velocity_loss_pct=round(loss_pct, 2),
        fatigue_risk=fat_pred.fatigue_risk,
        stop_probability=fat_pred.stop_probability,
        technique_anomaly=tech_pred.technique_anomaly_score,
        recommendation_action=rec.action,
        recommendation_reason=rec.reason,
        fatigue_model_status=fat_status,
        tech_model_status=tech_status,
        set_row_id=set_row_id,
    )
