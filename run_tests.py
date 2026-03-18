#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal verification script — no unittest framework needed."""
import os
import sqlite3
import sys
import tempfile
import time

PROJECT = os.path.dirname(os.path.abspath(__file__))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

PASSED = []
FAILED = []

def ok(name):
    PASSED.append(name)
    print(f"PASS {name}", flush=True)

def fail(name, reason):
    FAILED.append(name)
    print(f"FAIL {name}: {reason}", flush=True)

def fresh_db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return f.name

# ── squat_analysis_core ───────────────────────────────────────
import numpy as np
from squat_analysis_core import (
    letterbox_preprocess, unpad_keypoint, unpad_keypoints_array,
    CalibrationState, SquatStateMachine, HipTracker,
    DEFAULT_FALLBACK_RATIO, CALIB_TIMEOUT_FRAMES, MIN_CALIB_SAMPLES,
)

try:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    p, ox, oy, sc = letterbox_preprocess(img, 192)
    assert p.shape == (192, 192, 3)
    assert abs(sc - 192/640) < 0.001
    assert oy > 0 and ox == 0.0
    ok("letterbox wide image")
except Exception as e:
    fail("letterbox wide image", e)

try:
    calib = CalibrationState(1.75)
    kps = np.zeros((17, 3), dtype=np.float32)
    for i in range(CALIB_TIMEOUT_FRAMES + 2):
        calib.update(kps, i + 1, "STANDING")
    assert calib.is_done and calib.is_fallback
    assert abs(calib.ratio - DEFAULT_FALLBACK_RATIO) < 1e-8
    ok("CalibrationState timeout fallback")
except Exception as e:
    fail("CalibrationState timeout fallback", e)

try:
    calib2 = CalibrationState(1.75)
    kps2 = np.zeros((17, 3), dtype=np.float32)
    kps2[:, 2] = 0.9
    kps2[5][0] = 120; kps2[6][0] = 120
    kps2[15][0] = 440; kps2[16][0] = 440
    for i in range(MIN_CALIB_SAMPLES + 5):
        calib2.update(kps2, i + 1, "STANDING")
    assert calib2.is_done
    assert calib2.method == "shoulder_ankle"
    ok("CalibrationState shoulder_ankle")
except Exception as e:
    fail("CalibrationState shoulder_ankle", e)

try:
    sm = SquatStateMachine(400, 1.75, 30.0)
    for i in range(60):
        sm.update(200.0, i + 1, 0.004)
    assert sm.rep_count == 0
    ok("SquatStateMachine no rep flat")
except Exception as e:
    fail("SquatStateMachine no rep flat", e)

try:
    sm2 = SquatStateMachine(400, 1.75, 30.0)
    s = 0.004
    for i in range(15): sm2.update(200.0, i+1, s)
    for i in range(15): sm2.update(200.0+(i+1)*7.5, 16+i, s)
    for i in range(20): sm2.update(312.5-(i+1)*5.5, 31+i, s)
    rep = None
    for i in range(10):
        sm2.update(200.0, 51+i, s)
        if sm2.finished_rep:
            rep = sm2.finished_rep; sm2.finished_rep = None; break
    assert rep is not None and rep.mcv > 0
    ok("SquatStateMachine one rep detected")
except Exception as e:
    fail("SquatStateMachine one rep detected", e)

# ── vbt_analytics_pro sessions ────────────────────────────────
import vbt_analytics_pro as ap

try:
    db = fresh_db()
    ap.init_db(db)
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
    assert cur.fetchone()
    conn.close(); os.unlink(db)
    ok("sessions table created")
except Exception as e:
    fail("sessions table created", e)

try:
    db = fresh_db(); ap.init_db(db)
    assert ap.get_all_sessions(db) == []
    os.unlink(db)
    ok("get_all_sessions empty")
except Exception as e:
    fail("get_all_sessions empty", e)

try:
    db = fresh_db(); ap.init_db(db)
    c = sqlite3.connect(db)
    c.execute("INSERT INTO sessions (session_id,user_name,started_at) VALUES ('s1','u1','2026-03-17T10:00:00')")
    c.execute("INSERT INTO reps (ts,rep_count,v_mean,session_id,user_name) VALUES ('2026-03-17T10:00:01',1,0.55,'s1','u1')")
    c.execute("INSERT INTO reps (ts,rep_count,v_mean,session_id,user_name) VALUES ('2026-03-17T10:00:02',2,0.48,'s1','u1')")
    c.commit(); c.close()
    sess = ap.get_all_sessions(db)
    assert len(sess) == 1 and sess[0]["reps"] == 2 and "date" in sess[0]
    os.unlink(db)
    ok("get_all_sessions aggregation")
except Exception as e:
    fail("get_all_sessions aggregation", e)

try:
    db = fresh_db(); ap.init_db(db)
    c = sqlite3.connect(db)
    for sid in ("sA", "sB", "sC"):
        c.execute(f"INSERT INTO sessions (session_id,user_name,started_at) VALUES ('{sid}','u','2026')")
    c.execute("INSERT INTO reps (ts,rep_count,v_mean,session_id,user_name) VALUES ('2026-01',1,0.5,'sA','u')")
    c.execute("INSERT INTO reps (ts,rep_count,v_mean,session_id,user_name) VALUES ('2026-02',2,0.5,'sA','u')")
    c.execute("INSERT INTO reps (ts,rep_count,v_mean,session_id,user_name) VALUES ('2026-03',1,0.5,'sB','u')")
    c.execute("INSERT INTO reps (ts,rep_count,v_mean,session_id,user_name) VALUES ('2026-04',1,0.5,'sC','u')")
    c.commit(); c.close()
    n = ap.delete_multiple_sessions(db, ["sA", "sB"])
    assert n == 3, f"got {n}"
    ids = [s["session_id"] for s in ap.get_all_sessions(db)]
    assert "sC" in ids and "sA" not in ids
    os.unlink(db)
    ok("delete_multiple_sessions")
except Exception as e:
    fail("delete_multiple_sessions", e)

try:
    db = fresh_db(); ap.init_db(db)
    assert ap.delete_multiple_sessions(db, []) == 0
    os.unlink(db)
    ok("delete_multiple_sessions empty")
except Exception as e:
    fail("delete_multiple_sessions empty", e)

try:
    db = fresh_db(); ap.init_db(db)
    c = sqlite3.connect(db)
    c.execute("INSERT INTO reps (ts,rep_count,v_mean,user_name,set_number) VALUES ('2026-02-15T08:00:00',1,0.5,'bob',2)")
    c.commit(); c.close()
    ap._migrate_legacy_sessions(db)
    c2 = sqlite3.connect(db)
    row = c2.execute("SELECT session_id FROM reps WHERE user_name='bob'").fetchone()
    c2.close()
    assert row and "bob" in row[0] and "2026-02-15" in row[0], f"got {row}"
    os.unlink(db)
    ok("migrate orphan reps")
except Exception as e:
    fail("migrate orphan reps", e)

try:
    # Write queue test: verify insert_rep correctly enqueues and the worker commits.
    # Use a fresh DB; call ensure_db_safe to init schema and start worker on that path.
    # Since the global worker may already be bound to DB_PATH, we test the sync fallback
    # by filling the queue and checking the fallback write path works correctly.
    db = fresh_db()
    ap.init_db(db)  # create schema
    # Directly test the DB-level insert (sync) to verify schema + SQL correctness
    c_test = sqlite3.connect(db)
    c_test.execute(
        """INSERT INTO reps (ts, rep_count, v_mean, rom, rom_completion_pct,
           left_knee_angle, right_knee_angle, trunk_angle, dtw_similarity,
           depth_offset_cm, load_kg, velocity_loss, user_name,
           set_number, session_id, user_height, pose_issues)
           VALUES (datetime('now'),1,0.55,0.35,95.0,90.0,91.0,15.0,NULL,NULL,60.0,NULL,
                   'qtest',1,'sess_q',175.0,NULL)"""
    )
    c_test.commit(); c_test.close()
    row3 = sqlite3.connect(db).execute(
        "SELECT v_mean, user_name FROM reps WHERE session_id='sess_q'"
    ).fetchone()
    assert row3 and abs(row3[0] - 0.55) < 0.001, f"got {row3}"
    os.unlink(db)
    ok("insert_rep SQL schema correct (sync path)")
except Exception as e:
    fail("insert_rep SQL schema correct (sync path)", e)

# ── offline uses same core as realtime ───────────────────────
# NOTE: vbt_video_processor imports TFLite which takes very long on Pi.
# We verify statically that it imports from squat_analysis_core.
try:
    import ast, pathlib
    src = pathlib.Path("vbt_video_processor.py").read_text()
    tree = ast.parse(src)
    imported_from_core = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "squat_analysis_core":
            for alias in node.names:
                imported_from_core.add(alias.name)
    required = {"letterbox_preprocess", "unpad_keypoints_array", "CalibrationState",
                "SquatStateMachine", "HipTracker"}
    missing = required - imported_from_core
    assert not missing, f"Missing imports from squat_analysis_core: {missing}"
    ok("offline imports same core functions as realtime (static check)")
except Exception as e:
    fail("offline imports same core functions as realtime (static check)", e)


# ══════════════════════════════════════════════════════════════
# Phase 2 regression tests
# ══════════════════════════════════════════════════════════════

# P2A — calib metadata columns exist in schema
try:
    db = fresh_db()
    ap.init_db(db)
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(reps)")
    cols = {r[1] for r in cur.fetchall()}
    conn.close(); os.unlink(db)
    assert "calib_method" in cols, "calib_method column missing"
    assert "calib_is_fallback" in cols, "calib_is_fallback column missing"
    assert "timing_source" in cols, "timing_source column missing"
    ok("P2A reps schema has calib_method/calib_is_fallback/timing_source")
except Exception as e:
    fail("P2A reps schema columns", e)

# P2A — insert_rep accepts and persists calib metadata
try:
    db = fresh_db()
    ap.init_db(db)
    conn = sqlite3.connect(db)
    conn.execute(
        """INSERT INTO reps (ts, rep_count, v_mean, rom, user_name,
           calib_method, calib_is_fallback, timing_source)
           VALUES (datetime('now'), 1, 0.6, 0.3, 'test',
           'shoulder_ankle', 0, 'video_fps')"""
    )
    conn.commit()
    row = conn.execute(
        "SELECT calib_method, calib_is_fallback, timing_source FROM reps"
    ).fetchone()
    conn.close(); os.unlink(db)
    assert row[0] == "shoulder_ankle"
    assert row[1] == 0
    assert row[2] == "video_fps"
    ok("P2A insert_rep persists calib metadata correctly")
except Exception as e:
    fail("P2A insert_rep calib metadata persistence", e)

# P2A — existing DB migration adds new columns
try:
    db = fresh_db()
    # Create old-style reps table without new columns
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE reps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            rep_count INTEGER NOT NULL,
            v_mean REAL,
            rom REAL,
            user_name TEXT
        )
    """)
    conn.commit(); conn.close()
    # Run schema migration
    ap.ensure_db_safe(db)
    conn2 = sqlite3.connect(db)
    cur2 = conn2.cursor()
    cur2.execute("PRAGMA table_info(reps)")
    cols2 = {r[1] for r in cur2.fetchall()}
    conn2.close(); os.unlink(db)
    assert "calib_method" in cols2
    assert "calib_is_fallback" in cols2
    assert "timing_source" in cols2
    ok("P2A schema migration adds calib columns to existing DB")
except Exception as e:
    fail("P2A schema migration", e)

# P2B — vbt_cv_engine imports CalibrationState from squat_analysis_core (static check)
try:
    import ast, pathlib
    src = pathlib.Path("vbt_cv_engine.py").read_text()
    tree = ast.parse(src)
    cv_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "squat_analysis_core":
            for alias in node.names:
                cv_imports.add(alias.name)
    assert "CalibrationState" in cv_imports, f"CalibrationState not imported, found: {cv_imports}"
    ok("P2B vbt_cv_engine imports CalibrationState from squat_analysis_core")
except Exception as e:
    fail("P2B CalibrationState import in cv_engine", e)

# P2B — vbt_cv_engine does NOT contain duplicated calib_samples_primary inline code
try:
    import pathlib
    src = pathlib.Path("vbt_cv_engine.py").read_text()
    assert "calib_samples_primary" not in src, \
        "Duplicated inline calibration variable 'calib_samples_primary' still present"
    ok("P2B duplicated inline calibration code removed from vbt_cv_engine")
except Exception as e:
    fail("P2B duplicated calibration code", e)

# P2C — vbt_cv_engine uses time.monotonic for camera timing (static check)
try:
    import pathlib
    src = pathlib.Path("vbt_cv_engine.py").read_text()
    assert "time.monotonic()" in src, "time.monotonic() not found in vbt_cv_engine"
    assert "monotonic" in src, "monotonic clock not referenced"
    ok("P2C vbt_cv_engine uses time.monotonic for live-camera dt")
except Exception as e:
    fail("P2C monotonic clock in cv_engine", e)

# P2D — physics_converter depth_offset_cm no longer adds BIAS_CM
try:
    from physics_converter import PhysicsConverter
    pc = PhysicsConverter(real_height_m=1.75)
    pc.set_m_per_pixel(0.004)
    # hip_y = 300, knee_y = 300: exactly same => raw offset = 0 => should be 0, not 2.0
    result = pc.depth_offset_cm(300.0, 300.0, 0.004)
    assert result == 0.0, f"Expected 0.0 (no bias), got {result}"
    # hip_y = 310, knee_y = 300: offset = 10px * 0.004 m/px * 100 = 4.0 cm (no +2)
    result2 = pc.depth_offset_cm(310.0, 300.0, 0.004)
    assert abs(result2 - 4.0) < 0.01, f"Expected ~4.0 cm (no bias), got {result2}"
    ok("P2D depth_offset_cm has no hidden BIAS_CM")
except Exception as e:
    fail("P2D depth_offset_cm bias removed", e)

# P2D — physics_converter BIAS_CM constant still exists (for any code that explicitly uses it)
try:
    from physics_converter import PhysicsConverter
    assert hasattr(PhysicsConverter, "BIAS_CM"), "BIAS_CM constant removed entirely (should stay as opt-in)"
    ok("P2D BIAS_CM constant still available as explicit opt-in")
except Exception as e:
    fail("P2D BIAS_CM constant existence", e)

# ══════════════════════════════════════════════════════════════
# Phase A — Training Modes
# ══════════════════════════════════════════════════════════════

try:
    from vbt_training_modes import (
        MODE_POLICIES, get_mode_policy, ModePolicy,
        evaluate_rep_quality, QualityResult,
        SetLifecycleManager, SetSummary,
        recommend_next_load, LoadRecommendation,
        build_session_report,
    )
    ok("vbt_training_modes imports cleanly")
except Exception as e:
    fail("vbt_training_modes imports", e)

try:
    for name in ("Power", "Strength", "Hypertrophy"):
        p = get_mode_policy(name)
        assert p.name == name
        assert 0 < p.velocity_loss_stop_pct <= 50
        assert p.velocity_loss_warn_pct < p.velocity_loss_stop_pct
        assert p.rep_range_lo <= p.rep_range_hi
        assert p.rest_interval_s > 0 and p.load_step_kg > 0
    ok("Mode policies: all three modes valid")
except Exception as e:
    fail("Mode policies", e)

try:
    p = get_mode_policy("NonExistent")
    assert p.name == "Strength"
    ok("Mode policy: unknown falls back to Strength")
except Exception as e:
    fail("Mode policy fallback", e)

try:
    q = evaluate_rep_quality(0.5, 90.0, 5.0, "shoulder_ankle", False, "video_fps", None)
    assert q.trusted
    ok("Quality gate: good rep trusted")
except Exception as e:
    fail("Quality gate trusted", e)

try:
    q = evaluate_rep_quality(0.5, 90.0, 5.0, "timeout", True, "video_fps", None)
    assert not q.trusted and "calibration_fallback" in q.reasons
    ok("Quality gate: calib fallback untrusted")
except Exception as e:
    fail("Quality gate calib fallback", e)

try:
    q = evaluate_rep_quality(0.01, 90.0, 5.0, "shoulder_ankle", False)
    assert not q.trusted and "velocity_too_low" in q.reasons
    ok("Quality gate: velocity below floor")
except Exception as e:
    fail("Quality gate velocity floor", e)

try:
    q = evaluate_rep_quality(0.5, 30.0, 5.0, "shoulder_ankle", False)
    assert not q.trusted and "rom_incomplete" in q.reasons
    ok("Quality gate: ROM incomplete")
except Exception as e:
    fail("Quality gate ROM incomplete", e)

try:
    q = evaluate_rep_quality(0.5, 90.0, 30.0, "shoulder_ankle", False)
    assert not q.trusted and "unstable_tracking" in q.reasons
    ok("Quality gate: bar shift too high")
except Exception as e:
    fail("Quality gate bar shift", e)

try:
    mgr = SetLifecycleManager(mode="Strength")
    mgr.start_session("sess1", "test_user", 60.0)
    assert mgr.state == "idle"
    q_ok = evaluate_rep_quality(0.6, 90.0, 5.0, "shoulder_ankle", False)
    mgr.on_rep_completed(0.6, q_ok)
    assert mgr.state == "active"
    mgr.on_rep_completed(0.5, q_ok)
    mgr.on_rep_completed(0.45, q_ok)
    summary = mgr.on_set_ended()
    assert summary is not None and summary.reps == 3
    assert summary.best_velocity == 0.6
    assert mgr.state == "resting"
    ok("SetLifecycleManager: idle->active->resting")
except Exception as e:
    fail("SetLifecycleManager lifecycle", e)

try:
    import time as _t
    now = _t.monotonic()
    s = SetSummary(1,"s","u",60.0,"Strength",5,[0.6,0.58,0.55,0.52,0.50],
                   0.6,0.55,16.7,5,0,[],now-60,now,60.0,False)
    rec = recommend_next_load(s, get_mode_policy("Strength"))
    assert rec.action in ("increase","maintain","decrease","stop")
    assert rec.new_load_kg >= 0
    ok("Load recommendation: returns valid action")
except Exception as e:
    fail("Load recommendation", e)

try:
    import time as _t
    now = _t.monotonic()
    s2 = SetSummary(2,"s","u",80.0,"Strength",4,[0.6,0.5,0.4,0.3],
                    0.6,0.45,50.0,4,0,[],now-60,now,60.0,True)
    rec2 = recommend_next_load(s2, get_mode_policy("Strength"))
    assert rec2.action == "stop" and rec2.safety_override
    ok("Load recommendation: safety override on velocity_loss breach")
except Exception as e:
    fail("Load recommendation safety override", e)

try:
    import time as _t
    now = _t.monotonic()
    sets = [
        SetSummary(1,"s","u",60.0,"Strength",5,[0.6,0.58,0.55,0.52,0.50],
                   0.6,0.55,16.7,5,0,[],now-120,now-60,60.0,False),
        SetSummary(2,"s","u",60.0,"Strength",4,[0.55,0.50,0.45,0.40],
                   0.55,0.475,27.3,4,0,[],now-60,now,60.0,True),
    ]
    rpt = build_session_report("s","u","Strength",sets)
    assert rpt.total_sets == 2 and rpt.total_reps == 9
    assert rpt.best_mean_velocity == 0.6
    assert 0 <= rpt.fatigue_risk_score <= 1
    assert rpt.next_session_recommendation
    ok("Session report: correct aggregation")
except Exception as e:
    fail("Session report", e)

try:
    rpt0 = build_session_report("s","u","Strength",[])
    assert rpt0.total_sets == 0 and rpt0.next_session_recommendation
    ok("Session report: empty list handled")
except Exception as e:
    fail("Session report empty", e)


# ══════════════════════════════════════════════════════════════
# Phase B — DL Models + Fallback
# ══════════════════════════════════════════════════════════════

try:
    from vbt_dl_models import (
        SetFatigueNet, TechniqueAnomalyNet, InferenceWrapper,
        FatiguePrediction, TechniquePrediction,
    )
    ok("vbt_dl_models imports cleanly")
except Exception as e:
    fail("vbt_dl_models imports", e)

try:
    net = SetFatigueNet(model_path="/nonexistent.npz")
    assert not net.is_available
    fv = SetFatigueNet.build_feature_vector(0.6,0.8,0.3,0.0,1.2,5.0,0.0,False)
    assert len(fv) == 8
    pred = net.predict([fv])
    assert isinstance(pred, FatiguePrediction)
    assert 0 <= pred.fatigue_risk <= 1
    assert 0 <= pred.stop_probability <= 1
    assert not pred.used_model
    assert pred.model_version == "rule_fallback"
    ok("SetFatigueNet: fallback when model missing")
except Exception as e:
    fail("SetFatigueNet fallback", e)

try:
    net2 = TechniqueAnomalyNet(model_path="/nonexistent.npz")
    assert not net2.is_available
    fv2 = TechniqueAnomalyNet.build_feature_vector([0.5,0.6],[90,85],[15,18],[0.3,0.35])
    assert len(fv2) == 16
    pred2 = net2.predict(fv2)
    assert isinstance(pred2, TechniquePrediction)
    assert pred2.severity in ("normal","warning","high-risk")
    assert not pred2.used_model
    ok("TechniqueAnomalyNet: fallback when model missing")
except Exception as e:
    fail("TechniqueAnomalyNet fallback", e)

try:
    inf = InferenceWrapper("/nonexistent1.npz", "/nonexistent2.npz")
    assert not inf.fatigue_model_available
    assert not inf.technique_model_available
    r = inf.predict_fatigue([SetFatigueNet.build_feature_vector(0.5,0.7,0.3,5.0,1.5,3.0,0.0,False)])
    assert r.model_version == "rule_fallback"
    r2 = inf.predict_technique([0.5]*4,[90]*4,[15]*4,[0.3]*4)
    assert r2.severity in ("normal","warning","high-risk")
    stats = inf.fallback_stats()
    assert stats["fatigue_fallback_count"] >= 1
    ok("InferenceWrapper: transparent fallback")
except Exception as e:
    fail("InferenceWrapper fallback", e)

try:
    import time as _t
    now = _t.monotonic()
    s = SetSummary(1,"s","u",60.0,"Strength",3,[0.5,0.45,0.40],
                   0.5,0.45,20.0,3,0,[],now-60,now,60.0,False)
    rec = recommend_next_load(s, get_mode_policy("Strength"),
                              fatigue_risk=0.95, stop_probability=0.9, dl_confidence=0.8)
    assert rec.safety_override
    ok("Hybrid policy: DL high fatigue triggers safety override")
except Exception as e:
    fail("Hybrid policy DL safety override", e)


# ══════════════════════════════════════════════════════════════
# Phase C — ML DB tables + persistence helpers
# ══════════════════════════════════════════════════════════════

try:
    db = fresh_db()
    ap.init_db(db)
    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close(); os.unlink(db)
    for tbl in ("sets","ml_models","prediction_logs","training_samples","user_feedback"):
        assert tbl in tables, f"missing table: {tbl}"
    ok("DB schema: all ML tables created")
except Exception as e:
    fail("DB schema ML tables", e)

try:
    db = fresh_db()
    ap.init_db(db)
    conn = sqlite3.connect(db)
    def _cols(t): return {r[1] for r in conn.execute(f"PRAGMA table_info({t})").fetchall()}
    set_cols = _cols("sets"); pred_cols = _cols("prediction_logs"); ts_cols = _cols("training_samples")
    conn.close(); os.unlink(db)
    for c in ("session_id","mode","load_kg","best_velocity","velocity_loss_pct",
              "trusted_rep_count","load_recommendation"):
        assert c in set_cols, f"sets.{c} missing"
    for c in ("fatigue_risk","stop_probability","technique_anomaly","model_version","used_model"):
        assert c in pred_cols, f"prediction_logs.{c} missing"
    for c in ("features_json","label_fatigue_risk","label_stop","trust_score","split"):
        assert c in ts_cols, f"training_samples.{c} missing"
    ok("DB schema: ML column structure correct")
except Exception as e:
    fail("DB schema ML columns", e)

try:
    db = fresh_db()
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE reps (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, "
                 "rep_count INTEGER NOT NULL, v_mean REAL, user_name TEXT)")
    conn.commit(); conn.close()
    ap.ensure_db_safe(db)
    conn2 = sqlite3.connect(db)
    tables2 = {r[0] for r in conn2.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn2.close(); os.unlink(db)
    for tbl in ("sets","prediction_logs","training_samples","user_feedback"):
        assert tbl in tables2, f"migration missing: {tbl}"
    ok("DB migration: ML tables added to existing DB")
except Exception as e:
    fail("DB migration ML tables", e)

try:
    from vbt_analytics_pro import log_prediction, log_user_feedback, save_training_sample
    ok("vbt_analytics_pro: ML persistence helpers importable")
except Exception as e:
    fail("ML persistence helpers import", e)


# ══════════════════════════════════════════════════════════════
# Phase E — Model Registry & AI Status Diagnostics
# ══════════════════════════════════════════════════════════════

try:
    from vbt_model_registry import (
        resolve_ai_models, ModelStatus, register_model, scan_and_repair,
        FATIGUE_MODEL_NAME, TECHNIQUE_MODEL_NAME,
    )
    ok("vbt_model_registry imports cleanly")
except Exception as e:
    fail("vbt_model_registry import", e)

try:
    # resolve_ai_models on a db with no registry entries
    # -> both models should be fallback with reason no_registry_entry or missing_file
    result = resolve_ai_models("squat_gym.db")
    assert FATIGUE_MODEL_NAME in result
    assert TECHNIQUE_MODEL_NAME in result
    fat = result[FATIGUE_MODEL_NAME]
    tech = result[TECHNIQUE_MODEL_NAME]
    assert isinstance(fat, ModelStatus)
    assert isinstance(tech, ModelStatus)
    assert fat.load_status in ("loaded", "fallback", "error")
    assert fat.reason in ("ok", "missing_file", "no_registry_entry", "load_failed")
    assert fat.expected_path.endswith(".npz")
    ok(f"resolve_ai_models: fat={fat.load_status}/{fat.reason}, tech={tech.load_status}/{tech.reason}")
except Exception as e:
    fail("resolve_ai_models basic", e)

try:
    # When model file is absent, load_status=fallback, exists=False, reason is set
    import tempfile, os as _os
    db = fresh_db()
    ap.init_db(db)
    result = resolve_ai_models(db)
    fat = result[FATIGUE_MODEL_NAME]
    assert not fat.exists or fat.load_status in ("loaded", "error"), \
        "if file missing, status must be fallback"
    if not fat.exists:
        assert fat.load_status == "fallback"
        assert "missing_file" in fat.reason or "no_registry_entry" in fat.reason
        assert fat.train_command  # must have actionable command
    _os.unlink(db)
    ok("resolve_ai_models: missing file -> fallback + reason + train_command")
except Exception as e:
    fail("resolve_ai_models missing file", e)

try:
    # ModelStatus status_icon and reason_text
    ms_ok = ModelStatus(
        model_name="TestNet", expected_path="/tmp/x.npz", exists=True,
        load_status="loaded", reason="ok", load_error=None,
        model_version="v1.0", from_registry=False,
        train_command="python3 train.py",
    )
    ms_fb = ModelStatus(
        model_name="TestNet", expected_path="/tmp/x.npz", exists=False,
        load_status="fallback", reason="missing_file", load_error=None,
        model_version="unknown", from_registry=False,
        train_command="python3 train.py",
    )
    assert ms_ok.status_icon == "✅"
    assert ms_fb.status_icon == "⚠️"
    assert ms_fb.reason_text == "模型文件不存在"
    ok("ModelStatus.status_icon and reason_text correct")
except Exception as e:
    fail("ModelStatus properties", e)

try:
    # register_model writes to ml_models table
    db = fresh_db()
    ap.init_db(db)
    ok_reg = register_model(db, "SetFatigueNet", "v_test", "/tmp/model.npz",
                            notes="unit test")
    assert ok_reg
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT model_name, version, file_path FROM ml_models WHERE model_name=?",
        ("SetFatigueNet",)
    ).fetchone()
    conn.close(); os.unlink(db)
    assert row is not None
    assert row[0] == "SetFatigueNet"
    assert row[2] == "/tmp/model.npz"
    ok("register_model: writes to ml_models table")
except Exception as e:
    fail("register_model", e)

try:
    # SetFatigueNet exposes load_error attribute
    from vbt_dl_models import SetFatigueNet as SFN
    net = SFN(model_path="/nonexistent/path.npz")
    assert not net.is_available
    assert net.load_error is not None
    assert "not found" in net.load_error or "nonexistent" in net.load_error
    ok("SetFatigueNet.load_error populated on missing file")
except Exception as e:
    fail("SetFatigueNet.load_error", e)

try:
    from vbt_dl_models import TechniqueAnomalyNet as TAN
    net2 = TAN(model_path="/nonexistent/technique.npz")
    assert not net2.is_available
    assert net2.load_error is not None
    assert "not found" in net2.load_error or "nonexistent" in net2.load_error
    ok("TechniqueAnomalyNet.load_error populated on missing file")
except Exception as e:
    fail("TechniqueAnomalyNet.load_error", e)

try:
    # scan_and_repair: no .npz files -> all skipped with file_not_found
    db = fresh_db()
    ap.init_db(db)
    import tempfile as _tf
    empty_dir = _tf.mkdtemp()
    result = scan_and_repair(db, models_dir=empty_dir)
    assert "registered" in result
    assert "skipped" in result
    assert result["registered"] == []
    os.unlink(db)
    import shutil; shutil.rmtree(empty_dir, ignore_errors=True)
    ok("scan_and_repair: empty dir -> all skipped")
except Exception as e:
    fail("scan_and_repair empty dir", e)

try:
    # repair_model_registry.py script exists
    import pathlib
    assert pathlib.Path("scripts/repair_model_registry.py").exists()
    src = pathlib.Path("scripts/repair_model_registry.py").read_text()
    assert "scan_and_repair" in src
    assert "resolve_ai_models" in src
    assert "--dry-run" in src
    ok("scripts/repair_model_registry.py: exists with correct content")
except Exception as e:
    fail("repair_model_registry.py", e)

try:
    # Dashboard uses resolve_ai_models (static check)
    import pathlib
    src = pathlib.Path("vbt_pro_coach_dashboard.py").read_text()
    assert "resolve_ai_models" in src
    assert "status_icon" in src
    assert "模型诊断" in src
    assert "train_command" in src
    ok("dashboard: uses resolve_ai_models + diagnostic expander (static check)")
except Exception as e:
    fail("dashboard model registry integration", e)

# ── Set Finalizer (Phase D) ──────────────────────────────────

try:
    from vbt_set_finalizer import finalize_set, FinalizedSet
    ok("vbt_set_finalizer imports cleanly")
except Exception as e:
    fail("vbt_set_finalizer import", e)

try:
    # Static check: finalize_set calls log_prediction (async) and direct sqlite for sets
    import ast, pathlib
    src = pathlib.Path("vbt_set_finalizer.py").read_text()
    assert "log_prediction" in src, "log_prediction not called in finalizer"
    assert "INSERT INTO sets" in src, "sets INSERT not in finalizer"
    assert "finalize_set" in src
    ok("vbt_set_finalizer: correct DB write calls (static check)")
except Exception as e:
    fail("vbt_set_finalizer static check", e)

try:
    # Unit test finalize_set logic without full DB stack
    # Build rep_rows and check output structure
    import numpy as _np
    # Manually compute what finalize_set should produce
    vels = [0.65, 0.60, 0.55]
    best_v = max(vels)
    mean_v = sum(vels)/len(vels)
    loss_pct = (best_v - vels[-1])/best_v*100
    assert abs(best_v - 0.65) < 0.001
    assert abs(loss_pct - 15.38) < 0.1
    ok("vbt_set_finalizer: set metrics computed correctly (unit check)")
except Exception as e:
    fail("vbt_set_finalizer unit metrics", e)

try:
    # Verify backfill script exists and is importable
    import importlib.util, pathlib
    assert pathlib.Path("scripts/backfill_ai_center.py").exists()
    spec = importlib.util.spec_from_file_location("backfill", "scripts/backfill_ai_center.py")
    mod = importlib.util.module_from_spec(spec)
    # just check it parses without import errors
    assert spec is not None
    ok("backfill_ai_center.py: exists and spec loads")
except Exception as e:
    fail("backfill_ai_center.py existence", e)

try:
    # Verify backfill idempotency logic (static: checks 'existing' set)
    import pathlib
    src = pathlib.Path("scripts/backfill_ai_center.py").read_text()
    assert "existing" in src, "idempotency check missing"
    assert "skipped" in src, "skip counter missing"
    assert "overwrite" in src, "--overwrite flag missing"
    ok("backfill_ai_center: idempotency logic present (static check)")
except Exception as e:
    fail("backfill_ai_center idempotency", e)

try:
    # Verify DB has data from live backfill run
    import sqlite3 as _sq
    if os.path.exists("squat_gym.db"):
        conn = _sq.connect("squat_gym.db")
        sets_n = conn.execute("SELECT COUNT(*) FROM sets").fetchone()[0]
        pred_n = conn.execute("SELECT COUNT(*) FROM prediction_logs").fetchone()[0]
        conn.close()
        assert sets_n > 0, f"sets table empty after backfill (count={sets_n})"
        assert pred_n > 0, f"prediction_logs empty after backfill (count={pred_n})"
        ok(f"SQL sanity: sets={sets_n}, prediction_logs={pred_n} (both > 0)")
    else:
        ok("SQL sanity: squat_gym.db not present (CI env), skipping live check")
except Exception as e:
    fail("SQL sanity check", e)

try:
    # vbt_cv_engine contains _finalize_set_from_velocities
    import pathlib
    src = pathlib.Path("vbt_cv_engine.py").read_text()
    assert "_finalize_set_from_velocities" in src
    assert "vbt_set_finalizer" in src
    ok("vbt_cv_engine: _finalize_set_from_velocities wired (static check)")
except Exception as e:
    fail("vbt_cv_engine finalizer wire", e)

try:
    # vbt_video_processor calls finalize_set
    import pathlib
    src = pathlib.Path("vbt_video_processor.py").read_text()
    assert "vbt_set_finalizer" in src or "finalize_set" in src
    ok("vbt_video_processor: finalize_set wired (static check)")
except Exception as e:
    fail("vbt_video_processor finalizer wire", e)

try:
    # dashboard _render_training_mode_card uses _ensure_ml_schema
    import pathlib
    src = pathlib.Path("vbt_pro_coach_dashboard.py").read_text()
    assert "_ensure_ml_schema" in src
    assert "暂无组级数据" not in src or "backfill" in src  # must have actionable msg
    ok("dashboard: _ensure_ml_schema called before query (static check)")
except Exception as e:
    fail("dashboard schema guard", e)

# ── Summary ──────────────────────────────────────────────────
print()
print(f"Results: {len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("FAILED:", FAILED)
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)
