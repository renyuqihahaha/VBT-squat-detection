#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_models_end2end.py — One-shot training pipeline.

Usage:
    python3 scripts/train_models_end2end.py [--db squat_gym.db] [--out models/] [--force]

Tiers:
    production  : sets >= 50  (full confidence)
    bootstrap   : 15 <= sets < 50  (low confidence, use with care)
    none        : sets < 15  (skip, rule fallback)
"""
import argparse, json, os, sys, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vbt_analytics_pro import DB_PATH, init_db
from vbt_ml_pipeline import build_dataset, train_fatigue_net, train_technique_net
from vbt_model_registry import register_model, FATIGUE_MODEL_NAME, TECHNIQUE_MODEL_NAME


def run(db_path: str, out_dir: str, data_dir: str = "ml_data", force: bool = False) -> dict:
    print(f"\n{'='*60}")
    print(f"VBT End-to-End Training Pipeline")
    print(f"DB: {db_path}")
    print(f"Models: {out_dir}")
    print(f"{'='*60}\n")

    # Use init_db (no write-worker thread) to avoid hang in non-Streamlit context
    init_db(db_path)

    # ── 1. Count available sets ─────────────────────────────
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT COUNT(DISTINCT session_id || '|' || COALESCE(set_number,0))
            FROM reps WHERE v_mean IS NOT NULL AND v_mean > 0.02
        """)
        n_sets = cur.fetchone()[0] or 0
        cur.execute("""
            SELECT COUNT(DISTINCT session_id || '|' || COALESCE(set_number,0))
            FROM reps WHERE v_mean IS NOT NULL AND v_mean > 0.02
              AND (calib_is_fallback IS NULL OR calib_is_fallback = 0)
        """)
        n_high_trust = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM reps WHERE v_mean IS NOT NULL AND v_mean > 0.02")
        n_reps = cur.fetchone()[0] or 0
    finally:
        conn.close()

    print(f"Data summary:")
    print(f"  Total sets   : {n_sets}")
    print(f"  High-trust   : {n_high_trust}")
    print(f"  Total reps   : {n_reps}")

    if n_sets < 10 and not force:
        print(f"\n数据不足 ({n_sets} sets, 需要 ≥10)。")
        print("  使用 --force 强制训练 (不推荐)，或继续采集更多数据。")
        return {"status": "insufficient_data", "n_sets": n_sets}

    tier = "production" if n_sets >= 50 else "bootstrap"
    print(f"\nTraining tier: {tier.upper()}")
    if tier == "bootstrap":
        print("  ⚠️  Bootstrap 模式: 模型置信度低，仅供调试参考。")

    # ── 2. Build dataset ────────────────────────────────────
    print("\n[1/4] Building dataset...")
    stats = build_dataset(db_path, data_dir)
    print(f"  fatigue sets : {stats.get('fatigue_sets', 0)}")
    print(f"  technique reps: {stats.get('technique_reps', 0)}")

    results = {"tier": tier, "n_sets": n_sets, "n_high_trust": n_high_trust}

    # ── 3. Train SetFatigueNet ──────────────────────────────
    print("\n[2/4] Training SetFatigueNet...")
    epochs_fat = 40 if tier == "production" else 20
    fat_result = train_fatigue_net(data_dir, out_dir, epochs=epochs_fat)
    results["fatigue"] = fat_result
    if fat_result.get("error"):
        print(f"  SKIP: {fat_result.get('message', fat_result['error'])}")
    else:
        vpath = fat_result.get("versioned_path", "")
        version = fat_result.get("version", "unknown")
        print(f"  OK: {vpath} (tier={fat_result.get('training_tier')})")
        print(f"      val_bce={fat_result.get('val_bce', 'N/A'):.4f}  "
              f"train={fat_result.get('n_train', 0)}  val={fat_result.get('n_val', 0)}")
        # Register (allow_overwrite=False prevents accidental clobber)
        ok = register_model(
            db_path, FATIGUE_MODEL_NAME, version, vpath,
            train_samples=fat_result.get("n_train"),
            val_metric=fat_result.get("val_bce"),
            notes=f"tier={tier}",
            allow_overwrite=False, set_active=True,
        )
        print(f"  Registry: {'OK' if ok else 'FAILED'}")
        results["fatigue_registered"] = ok

    # ── 4. Train TechniqueAnomalyNet ────────────────────────
    print("\n[3/4] Training TechniqueAnomalyNet...")
    epochs_tech = 60 if tier == "production" else 30
    tech_result = train_technique_net(data_dir, out_dir, epochs=epochs_tech)
    results["technique"] = tech_result
    if tech_result.get("error"):
        print(f"  SKIP: {tech_result.get('message', tech_result['error'])}")
    else:
        tpath = tech_result.get("versioned_path", "")
        tversion = tech_result.get("version", "unknown")
        print(f"  OK: {tpath} (tier={tech_result.get('training_tier')})")
        print(f"      val_mse={tech_result.get('val_mse', 'N/A'):.6f}  "
              f"baseline={tech_result.get('baseline_error', 'N/A'):.4f}  "
              f"train={tech_result.get('n_train', 0)}  val={tech_result.get('n_val', 0)}")
        ok2 = register_model(
            db_path, TECHNIQUE_MODEL_NAME, tversion, tpath,
            train_samples=tech_result.get("n_train"),
            val_metric=tech_result.get("val_mse"),
            notes=f"tier={tier}",
            allow_overwrite=False, set_active=True,
        )
        print(f"  Registry: {'OK' if ok2 else 'FAILED'}")
        results["technique_registered"] = ok2

    # ── 5. SQL verification ─────────────────────────────────
    print("\n[4/4] SQL verification...")
    import time; time.sleep(1.0)  # allow async write queue to flush
    conn2 = sqlite3.connect(db_path)
    sets_n = conn2.execute("SELECT COUNT(*) FROM sets").fetchone()[0]
    pred_n = conn2.execute("SELECT COUNT(*) FROM prediction_logs").fetchone()[0]
    reg_n  = conn2.execute("SELECT COUNT(*) FROM ml_models").fetchone()[0]
    try:
        active_rows = conn2.execute(
            "SELECT key, value FROM system_config WHERE key LIKE 'active_%'"
        ).fetchall()
    except Exception:
        active_rows = []
    conn2.close()

    print(f"  sets              : {sets_n}")
    print(f"  prediction_logs   : {pred_n}")
    print(f"  ml_models registry: {reg_n}")
    for k, v in active_rows:
        print(f"  {k}: {v}")

    results["sql"] = {
        "sets": sets_n, "prediction_logs": pred_n,
        "ml_models": reg_n, "active_models": dict(active_rows),
    }

    print(f"\n{'='*60}")
    print(f"Training complete. Tier: {tier}")
    print(f"{'='*60}\n")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--out", default="models")
    p.add_argument("--data", default="ml_data")
    p.add_argument("--force", action="store_true", help="Train even if sets < 15")
    args = p.parse_args()
    run(args.db, args.out, args.data, force=args.force)
