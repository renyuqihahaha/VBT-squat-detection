#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/backfill_ai_center.py — Backfill sets + prediction_logs from existing reps.

Usage:
    python3 scripts/backfill_ai_center.py [--db squat_gym.db] [--dry-run] [--overwrite]

Idempotent: skips (session_id, set_number) pairs already in `sets` unless --overwrite.
"""
import argparse, os, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vbt_analytics_pro import DB_PATH, ensure_db_safe
from vbt_set_finalizer import finalize_set


def backfill(db_path: str, dry_run: bool = False, overwrite: bool = False) -> dict:
    ensure_db_safe(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT session_id, set_number,
               COALESCE(user_name,'qiqi') AS user_name,
               COALESCE(load_kg,0) AS load_kg,
               v_mean, rom, rom_completion_pct, velocity_loss,
               calib_is_fallback, calib_method, timing_source,
               pose_issues, left_knee_angle, right_knee_angle, trunk_angle
        FROM reps
        WHERE v_mean IS NOT NULL AND v_mean > 0.02
        ORDER BY session_id, set_number, ts
    """)
    rows = cur.fetchall()
    groups: dict = {}
    for r in rows:
        key = (r["session_id"] or "unknown", r["set_number"] or 1)
        groups.setdefault(key, {"reps": [], "user_name": r["user_name"],
                                "load_kg": r["load_kg"]})
        groups[key]["reps"].append(dict(r))

    existing: set = set()
    if not overwrite:
        try:
            cur.execute("SELECT session_id, set_number FROM sets")
            existing = {(r[0], r[1]) for r in cur.fetchall()}
        except Exception:
            pass
    conn.close()

    n_written = n_skipped = n_failed = 0
    for (sid, sno), info in groups.items():
        if (sid, sno) in existing:
            n_skipped += 1
            continue
        rep_rows = info["reps"]
        if dry_run:
            vels = [float(r["v_mean"]) for r in rep_rows]
            best = max(vels) if vels else 0
            print(f"[DRY-RUN] session={sid[:24]} set={sno} reps={len(rep_rows)} best={best:.3f}")
            n_written += 1
            continue
        try:
            result = finalize_set(
                session_id=str(sid),
                set_number=int(sno),
                user_name=info["user_name"],
                mode="Strength",
                load_kg=float(info["load_kg"]),
                rep_rows=rep_rows,
                db_path=db_path,
            )
            if result:
                print(f"  OK session={str(sid)[:24]} set={sno} reps={result.reps} "
                      f"best={result.best_velocity:.3f} rec={result.recommendation_action} "
                      f"model={result.fatigue_model_status}")
                n_written += 1
            else:
                print(f"  WARN session={sid} set={sno}: finalize returned None")
                n_failed += 1
        except Exception as e:
            print(f"  ERR session={sid} set={sno}: {e}")
            n_failed += 1

    # Wait for async writes to flush
    import time; time.sleep(1.0)

    conn2 = sqlite3.connect(db_path)
    sets_count = conn2.execute("SELECT COUNT(*) FROM sets").fetchone()[0]
    pred_count  = conn2.execute("SELECT COUNT(*) FROM prediction_logs").fetchone()[0]
    conn2.close()

    print(f"\nBackfill done: written={n_written} skipped={n_skipped} failed={n_failed}")
    print(f"DB: sets={sets_count} prediction_logs={pred_count}")
    return {"written": n_written, "skipped": n_skipped, "failed": n_failed,
            "sets_count": sets_count, "pred_count": pred_count}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    backfill(args.db, dry_run=args.dry_run, overwrite=args.overwrite)
