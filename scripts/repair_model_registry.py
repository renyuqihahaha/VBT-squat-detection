#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/repair_model_registry.py — Scan models/ and fix ml_models registry.

Usage:
    python3 scripts/repair_model_registry.py [--db squat_gym.db] [--models-dir models/] [--dry-run]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vbt_analytics_pro import DB_PATH
from vbt_model_registry import resolve_ai_models, scan_and_repair, MODELS_DIR


def main():
    p = argparse.ArgumentParser(description="Scan models/ and repair ml_models registry")
    p.add_argument("--db", default=DB_PATH, help="Path to SQLite DB")
    p.add_argument("--models-dir", default=MODELS_DIR, help="Path to models directory")
    p.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = p.parse_args()

    print(f"DB: {args.db}")
    print(f"Models dir: {args.models_dir}")
    print()

    # Show current status before repair
    print("=== Current model status ===")
    statuses = resolve_ai_models(args.db)
    for name, st in statuses.items():
        print(f"  {st.status_icon} {name}")
        print(f"     status      : {st.load_status}")
        print(f"     reason      : {st.reason_text}")
        print(f"     path        : {st.expected_path}")
        print(f"     file_exists : {st.exists}")
        print(f"     version     : {st.model_version}")
        print(f"     from_registry: {st.from_registry}")
        if st.load_error:
            print(f"     error       : {st.load_error}")
        if st.load_status != "loaded":
            print(f"     fix command : {st.train_command}")
        print()

    if args.dry_run:
        print("[DRY-RUN] No changes written.")
        return

    # Repair
    print("=== Running repair ===")
    result = scan_and_repair(args.db, args.models_dir)
    for r in result["registered"]:
        print(f"  REGISTERED: {r['model']} -> {r['path']}")
    for s in result["skipped"]:
        print(f"  SKIPPED: {s['model']} ({s['reason']}) {s['path']}")

    # Show updated status after repair
    print()
    print("=== Status after repair ===")
    statuses2 = resolve_ai_models(args.db)
    for name, st in statuses2.items():
        print(f"  {st.status_icon} {name}: {st.load_status} ({st.reason_text})")
        if st.load_status != "loaded":
            print(f"     Train: {st.train_command}")


if __name__ == "__main__":
    main()
