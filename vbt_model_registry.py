#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vbt_model_registry.py — Model discovery, status resolution, and registry management.

Primary API:
    resolve_ai_models(db_path=None) -> dict[str, ModelStatus]

Status resolution priority (per model):
    1. DB model_registry table (active entry)
    2. Default models/ directory
    3. Missing -> fallback with explicit reason
"""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("vbt_model_registry")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Canonical model identifiers
FATIGUE_MODEL_NAME = "SetFatigueNet"
TECHNIQUE_MODEL_NAME = "TechniqueAnomalyNet"

_DEFAULTS = {
    FATIGUE_MODEL_NAME: "set_fatigue_net.npz",
    TECHNIQUE_MODEL_NAME: "technique_anomaly_net.npz",
}

_TRAIN_COMMANDS = {
    FATIGUE_MODEL_NAME: (
        "python3 vbt_ml_pipeline.py build-dataset --db squat_gym.db --out ml_data/ && "
        "python3 vbt_ml_pipeline.py train-fatigue --data ml_data/ --out models/"
    ),
    TECHNIQUE_MODEL_NAME: (
        "python3 vbt_ml_pipeline.py build-dataset --db squat_gym.db --out ml_data/ && "
        "python3 vbt_ml_pipeline.py train-technique --data ml_data/ --out models/"
    ),
}


@dataclass
class ModelStatus:
    model_name: str
    expected_path: str          # resolved absolute path we searched
    exists: bool                # whether the file is present on disk
    load_status: str            # "loaded" | "fallback" | "error"
    reason: str                 # "ok" | "missing_file" | "load_failed" | "no_registry_entry"
    load_error: Optional[str]   # exception message if load_failed
    model_version: str          # version string from registry or file, or "unknown"
    from_registry: bool         # True if path came from DB registry
    train_command: str          # actionable command to create this model
    extra: dict = field(default_factory=dict)

    @property
    def status_icon(self) -> str:
        return {"loaded": "✅", "fallback": "⚠️", "error": "❌"}.get(self.load_status, "❓")

    @property
    def reason_text(self) -> str:
        texts = {
            "ok": "已加载",
            "missing_file": "模型文件不存在",
            "load_failed": "文件存在但加载失败",
            "no_registry_entry": "Registry 无记录，使用默认路径",
            "disabled": "已禁用",
        }
        return texts.get(self.reason, self.reason)


def _get_registry_path(db_path: str, model_name: str) -> Optional[tuple[str, str]]:
    """
    Query ml_models table for the latest active entry for model_name.
    Returns (file_path, version) or None.
    """
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT file_path, version FROM ml_models "
            "WHERE model_name=? AND file_path IS NOT NULL "
            "ORDER BY trained_at DESC, id DESC LIMIT 1",
            (model_name,),
        )
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            return (str(row[0]), str(row[1] or "unknown"))
    except Exception as e:
        logger.debug("Registry lookup failed for %s: %s", model_name, e)
    return None


def _default_path(model_name: str) -> str:
    fname = _DEFAULTS.get(model_name, model_name.lower() + ".npz")
    return os.path.join(MODELS_DIR, fname)


def _probe_model(model_name: str, path: str) -> tuple[bool, Optional[str]]:
    """
    Attempt a lightweight load probe of the .npz file.
    Returns (success, error_message).
    """
    import numpy as np
    try:
        data = np.load(path, allow_pickle=False)
        _ = data.files  # force read
        return True, None
    except Exception as e:
        return False, str(e)


def resolve_ai_models(db_path: Optional[str] = None) -> dict[str, ModelStatus]:
    """
    Resolve status for all AI models.
    Priority: DB registry > default models/ path.
    Always returns a complete status dict even if both sources fail.
    """
    from vbt_analytics_pro import DB_PATH as _default_db
    _db = db_path or _default_db
    results: dict[str, ModelStatus] = {}

    for model_name in (FATIGUE_MODEL_NAME, TECHNIQUE_MODEL_NAME):
        train_cmd = _TRAIN_COMMANDS.get(model_name, "# see vbt_ml_pipeline.py --help")

        # 1. Try registry
        reg = _get_registry_path(_db, model_name)
        if reg:
            reg_path, reg_version = reg
            resolved_path = reg_path if os.path.isabs(reg_path) else os.path.join(
                os.path.dirname(os.path.abspath(__file__)), reg_path
            )
            from_registry = True
        else:
            resolved_path = _default_path(model_name)
            reg_version = "unknown"
            from_registry = False

        exists = os.path.isfile(resolved_path)

        if not exists:
            results[model_name] = ModelStatus(
                model_name=model_name,
                expected_path=resolved_path,
                exists=False,
                load_status="fallback",
                reason="missing_file" if from_registry else "no_registry_entry",
                load_error=None,
                model_version="unknown",
                from_registry=from_registry,
                train_command=train_cmd,
            )
            logger.info("%s: file not found at %s", model_name, resolved_path)
            continue

        # 2. Probe load
        success, err = _probe_model(model_name, resolved_path)
        if success:
            results[model_name] = ModelStatus(
                model_name=model_name,
                expected_path=resolved_path,
                exists=True,
                load_status="loaded",
                reason="ok",
                load_error=None,
                model_version=reg_version,
                from_registry=from_registry,
                train_command=train_cmd,
            )
            logger.info("%s: loaded from %s (version=%s)", model_name, resolved_path, reg_version)
        else:
            results[model_name] = ModelStatus(
                model_name=model_name,
                expected_path=resolved_path,
                exists=True,
                load_status="error",
                reason="load_failed",
                load_error=err,
                model_version=reg_version,
                from_registry=from_registry,
                train_command=train_cmd,
            )
            logger.warning("%s: file exists but load failed: %s", model_name, err)

    return results


def register_model(
    db_path: str,
    model_name: str,
    version: str,
    file_path: str,
    model_type: str = "npz",
    train_samples: Optional[int] = None,
    val_metric: Optional[float] = None,
    notes: Optional[str] = None,
    allow_overwrite: bool = False,
    set_active: bool = True,
) -> bool:
    """
    Insert or update a model entry in ml_models registry.

    allow_overwrite=False (default): if (model_name, version) already exists,
        auto-generate a new version by appending '_b' suffix to avoid overwrite.
    allow_overwrite=True: update the existing record in place.
    set_active=True: write an 'active_{model_name}' key to system_config.

    Returns True on success.
    """
    try:
        conn = sqlite3.connect(db_path)
        if not allow_overwrite:
            # Check if (model_name, version) already exists
            existing = conn.execute(
                "SELECT id FROM ml_models WHERE model_name=? AND version=?",
                (model_name, version),
            ).fetchone()
            if existing:
                # Auto-generate a non-conflicting version
                version = f"{version}_b{existing[0]}"
                logger.info(
                    "register_model: version conflict, using auto-version '%s' for %s",
                    version, model_name,
                )
        conn.execute(
            """
            INSERT INTO ml_models (model_name, version, model_type, file_path,
                                   trained_at, train_samples, val_metric, notes)
            VALUES (?, ?, ?, ?, datetime('now'), ?, ?, ?)
            ON CONFLICT(model_name, version) DO UPDATE SET
                file_path=excluded.file_path,
                trained_at=excluded.trained_at,
                train_samples=excluded.train_samples,
                val_metric=excluded.val_metric,
                notes=excluded.notes
            """,
            (model_name, version, model_type, file_path, train_samples, val_metric, notes),
        )
        conn.commit()
        logger.info("Registered %s v%s -> %s", model_name, version, file_path)

        if set_active:
            _set_active_model(conn, model_name, version, file_path)
            conn.commit()

        conn.close()
        return True
    except Exception as e:
        logger.error("register_model failed: %s", e)
        return False


def _set_active_model(conn: sqlite3.Connection, model_name: str, version: str, file_path: str) -> None:
    """
    Persist active model pointer in system_config table (key=active_{model_name}).
    Creates the table if it doesn't exist.
    """
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_config (
                key   TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        import json as _json
        payload = _json.dumps({"version": version, "file_path": file_path})
        conn.execute(
            """
            INSERT INTO system_config(key, value, updated_at)
            VALUES(?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (f"active_{model_name}", payload),
        )
        logger.debug("Active model pointer updated: active_%s -> v%s", model_name, version)
    except Exception as e:
        logger.warning("_set_active_model failed: %s", e)


def get_active_model(db_path: str, model_name: str) -> Optional[dict]:
    """
    Return active model info dict {version, file_path} or None.
    """
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT value FROM system_config WHERE key=?",
            (f"active_{model_name}",),
        ).fetchone()
        conn.close()
        if row:
            import json as _json
            return _json.loads(row[0])
    except Exception:
        pass
    return None


def scan_and_repair(db_path: str, models_dir: Optional[str] = None) -> dict:
    """
    Scan models_dir for known .npz files, register any that are missing from registry.
    Returns summary dict.
    """
    from vbt_analytics_pro import init_db
    init_db(db_path)  # safe: idempotent, no background thread
    mdir = models_dir or MODELS_DIR
    registered = []
    skipped = []

    for model_name, fname in _DEFAULTS.items():
        fpath = os.path.join(mdir, fname)
        if not os.path.isfile(fpath):
            skipped.append({"model": model_name, "reason": "file_not_found", "path": fpath})
            continue
        # Check if already in registry
        reg = _get_registry_path(db_path, model_name)
        if reg and os.path.normpath(reg[0]) == os.path.normpath(fpath):
            skipped.append({"model": model_name, "reason": "already_registered", "path": fpath})
            continue
        ok = register_model(
            db_path, model_name, version="scanned",
            file_path=fpath, notes="auto-registered by repair script"
        )
        if ok:
            registered.append({"model": model_name, "path": fpath})
        else:
            skipped.append({"model": model_name, "reason": "register_failed", "path": fpath})

    return {"registered": registered, "skipped": skipped}
