#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBT 分析核心：几何计算、数据库、审计日志。

重构变更 (Phase 1):
  B. sessions 表：真正按 session_id 聚合；兼容旧数据迁移。
  C. 单写线程队列：schema 初始化只做一次；insert_rep 不再每次 CREATE TABLE。
"""

import logging
import os
import queue
import sqlite3
import threading
from datetime import datetime
from typing import Optional

import numpy as np

from physics_converter import calculate_m_per_pixel, get_depth_offset, DEFAULT_SCALE

logger = logging.getLogger("vbt_analytics_pro")

MODEL_PATH = "models/movenet_lightning.tflite"
DB_PATH = "squat_gym.db"
CONF_THRESHOLD = 0.25


def angle_deg(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None:
        return None
    v1 = np.array([p1[1] - p2[1], p1[0] - p2[0]], dtype=float)
    v2 = np.array([p3[1] - p2[1], p3[0] - p2[0]], dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def trunk_angle_deg(shoulder_mid, hip_mid):
    if shoulder_mid is None or hip_mid is None:
        return None
    dx = shoulder_mid[1] - hip_mid[1]
    dy = shoulder_mid[0] - hip_mid[0]
    norm = np.sqrt(dx * dx + dy * dy)
    if norm < 1e-6:
        return None
    c = np.clip(-dy / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def dtw_distance(seq_a, seq_b):
    a = np.asarray(seq_a, dtype=float)
    b = np.asarray(seq_b, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    if a.shape[1] > 1:
        a = np.mean(a, axis=1, keepdims=True)
    if b.shape[1] > 1:
        b = np.mean(b, axis=1, keepdims=True)
    n, m = len(a), len(b)
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1, 0] - b[j - 1, 0])
            d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[n, m])


def dtw_similarity(seq_current, seq_standard):
    if not seq_current or not seq_standard:
        return None
    dist = dtw_distance(seq_current, seq_standard)
    norm = max(len(seq_current), len(seq_standard), 1)
    return float(1.0 / (1.0 + dist / norm))


def _ensure_reps_schema(conn):
    cur = conn.cursor()
    # Guard: table may not exist yet on a brand-new DB opened via ensure_db_safe
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reps'")
    if not cur.fetchone():
        return
    cur.execute("PRAGMA table_info(reps)")
    cols = {r[1] for r in cur.fetchall()}
    expected = [
        ("rom", "REAL"),
        ("rom_completion_pct", "REAL"),
        ("dtw_similarity", "REAL"),
        ("depth_offset_cm", "REAL"),
        ("load_kg", "REAL"),
        ("velocity_loss", "REAL"),
        ("user_id", "INTEGER"),
        ("user_name", "TEXT"),
        ("set_number", "INTEGER"),
        ("session_id", "TEXT"),
        ("user_height", "REAL"),
        ("pose_issues", "TEXT"),
        ("calib_method", "TEXT"),
        ("calib_is_fallback", "INTEGER"),
        ("timing_source", "TEXT"),
    ]
    for name, t in expected:
        if name not in cols:
            try:
                cur.execute(f"ALTER TABLE reps ADD COLUMN {name} {t}")
            except sqlite3.OperationalError:
                pass
    cur.execute("UPDATE reps SET user_id = 1 WHERE user_id IS NULL")
    cur.execute("UPDATE reps SET user_name = 'qiqi' WHERE user_name IS NULL OR user_name = ''")
    conn.commit()


def _ensure_analysis_logs_schema(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_logs'")
    if not cur.fetchone():
        return
    cur.execute("PRAGMA table_info(analysis_logs)")
    cols = {r[1] for r in cur.fetchall()}
    if "user_name" not in cols:
        try:
            cur.execute("ALTER TABLE analysis_logs ADD COLUMN user_name TEXT")
            cur.execute("UPDATE analysis_logs SET user_name = 'qiqi' WHERE user_name IS NULL OR user_name = ''")
        except sqlite3.OperationalError:
            pass
    conn.commit()


def _enable_wal(conn: sqlite3.Connection) -> None:
    """开启 WAL 模式，提升并发写入稳定性（幂等）。"""
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass


def _ensure_sessions_schema(conn: sqlite3.Connection) -> None:
    """确保 sessions 表存在（兼容旧库）。"""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            user_name    TEXT NOT NULL DEFAULT 'qiqi',
            started_at   TEXT,
            ended_at     TEXT,
            source_type  TEXT,
            source_name  TEXT,
            notes        TEXT
        )
        """
    )
    conn.commit()


def _ensure_ml_schema(conn: sqlite3.Connection) -> None:
    """确保 ML lifecycle 相关表存在（幂等，向后兼容）。"""
    cur = conn.cursor()
    # 组级元数据
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sets (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            user_name   TEXT NOT NULL DEFAULT 'qiqi',
            set_number  INTEGER NOT NULL,
            mode        TEXT,
            load_kg     REAL,
            reps        INTEGER,
            best_velocity REAL,
            mean_velocity REAL,
            velocity_loss_pct REAL,
            trusted_rep_count INTEGER,
            untrusted_rep_count INTEGER,
            quality_reasons TEXT,
            started_at  TEXT,
            ended_at    TEXT,
            was_stopped_by_policy INTEGER DEFAULT 0,
            load_recommendation TEXT,
            recommendation_reason TEXT
        )
    """)
    # 模型注册表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_models (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name  TEXT NOT NULL,
            version     TEXT NOT NULL,
            model_type  TEXT,
            file_path   TEXT,
            trained_at  TEXT,
            train_samples INTEGER,
            val_metric  REAL,
            notes       TEXT,
            UNIQUE(model_name, version)
        )
    """)
    # 预测日志
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            session_id  TEXT,
            set_id      INTEGER,
            model_name  TEXT,
            model_version TEXT,
            prediction_type TEXT,
            input_hash  TEXT,
            fatigue_risk REAL,
            stop_probability REAL,
            technique_anomaly REAL,
            confidence  REAL,
            used_model  INTEGER DEFAULT 0,
            recommendation TEXT,
            recommendation_reason TEXT
        )
    """)
    # 训练样本（特征 + 标签 + 信任元数据）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_samples (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            set_id      INTEGER,
            user_name   TEXT,
            created_at  TEXT,
            features_json TEXT NOT NULL,
            label_fatigue_risk REAL,
            label_stop  INTEGER,
            label_source TEXT DEFAULT 'auto',
            trust_score REAL DEFAULT 1.0,
            quality_reasons TEXT,
            split       TEXT DEFAULT 'unassigned'
        )
    """)
    # 用户反馈标签
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            session_id  TEXT,
            set_id      INTEGER,
            user_name   TEXT,
            rpe         REAL,
            felt_label  TEXT,
            notes       TEXT
        )
    """)
    conn.commit()


def _migrate_legacy_sessions(db_path: str) -> None:
    """
    将 reps 表中尚未在 sessions 表中登记的 session_id 补录进去。
    兼容旧数据：
      - 已有 session_id 字段且非空 → 直接补录
      - session_id 为空 → 用 "user_name__date__setN" 生成兼容 ID，并回写 reps.session_id
    保守策略：只 INSERT OR IGNORE，不修改已有 sessions 记录。
    """
    if not os.path.exists(db_path):
        return
    try:
        conn = sqlite3.connect(db_path)
        _enable_wal(conn)
        cur = conn.cursor()
        # 检查 reps 表是否存在
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reps'")
        if not cur.fetchone():
            conn.close()
            return

        # 1. 补录已有 session_id 但不在 sessions 表中的行
        cur.execute(
            """
            SELECT DISTINCT r.session_id,
                   COALESCE(r.user_name, 'qiqi'),
                   MIN(r.ts)
            FROM reps r
            LEFT JOIN sessions s ON s.session_id = r.session_id
            WHERE r.session_id IS NOT NULL
              AND r.session_id != ''
              AND s.session_id IS NULL
            GROUP BY r.session_id
            """
        )
        rows_existing = cur.fetchall()
        for sid, uname, started_at in rows_existing:
            cur.execute(
                "INSERT OR IGNORE INTO sessions (session_id, user_name, started_at, source_type) VALUES (?, ?, ?, ?)",
                (sid, uname or "qiqi", started_at, "migrated"),
            )

        # 2. 为 session_id 为空的 reps 生成兼容 session_id
        cur.execute(
            """
            SELECT id, COALESCE(user_name,'qiqi'), ts, COALESCE(set_number, 1)
            FROM reps
            WHERE session_id IS NULL OR session_id = ''
            ORDER BY ts
            """
        )
        orphan_rows = cur.fetchall()
        for row_id, uname, ts, set_no in orphan_rows:
            date_str = str(ts)[:10] if ts else "1970-01-01"
            compat_sid = f"{uname}__{date_str}__set{set_no}"
            cur.execute("UPDATE reps SET session_id = ? WHERE id = ?", (compat_sid, row_id))
            cur.execute(
                "INSERT OR IGNORE INTO sessions (session_id, user_name, started_at, source_type) VALUES (?, ?, ?, ?)",
                (compat_sid, uname, ts, "migrated_compat"),
            )

        conn.commit()
        conn.close()
        if rows_existing or orphan_rows:
            logger.info(
                "sessions 迁移完成: %d 条已有 session_id 补录, %d 条孤立 reps 生成兼容 ID",
                len(rows_existing), len(orphan_rows),
            )
    except Exception as e:
        logger.warning("sessions 迁移失败（非致命）: %s", e)


# ──────────────────────────────────────────────────────────────
# 单写线程队列（C 部分）
# ──────────────────────────────────────────────────────────────

_write_queue: queue.Queue = queue.Queue(maxsize=500)
_write_worker_started = False
_write_worker_lock = threading.Lock()
_write_worker_db_path: str = DB_PATH  # tracks which db the active worker is serving


def _start_write_worker(db_path: str) -> None:
    """启动单写线程（幂等，只启动一次）。"""
    global _write_worker_started, _write_worker_db_path
    with _write_worker_lock:
        if _write_worker_started:
            return
        _write_worker_started = True
        _write_worker_db_path = db_path

    def _worker() -> None:
        conn: Optional[sqlite3.Connection] = None
        while True:
            try:
                item = _write_queue.get(timeout=5)
                if item is None:  # 毒丸，退出
                    break
                sql, params = item
                if conn is None:
                    conn = sqlite3.connect(db_path, check_same_thread=False)
                    _enable_wal(conn)
                try:
                    conn.execute(sql, params)
                    conn.commit()
                except sqlite3.OperationalError as e:
                    logger.warning("写队列执行失败: %s | sql=%s", e, sql[:80])
                finally:
                    _write_queue.task_done()
            except queue.Empty:
                # 空闲时提交 WAL checkpoint，避免 WAL 文件膨胀
                if conn is not None:
                    try:
                        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    except Exception:
                        pass
            except Exception as e:
                logger.error("写线程异常: %s", e)

    t = threading.Thread(target=_worker, daemon=True, name="vbt-db-writer")
    t.start()
    logger.info("数据库单写线程已启动 (db=%s)", db_path)


def _enqueue_write(sql: str, params: tuple) -> None:
    """将写操作投入队列；队列满时同步回退写入（避免丢数据）。"""
    try:
        _write_queue.put_nowait((sql, params))
    except queue.Full:
        logger.warning("写队列已满，同步回退写入")
        try:
            conn = sqlite3.connect(_write_worker_db_path)
            _enable_wal(conn)
            conn.execute(sql, params)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("同步回退写入失败: %s", e)


def ensure_db_safe(db_path: str = DB_PATH) -> None:
    """
    断点续存：仅当 DB 文件不存在时建表；已存在则仅连接并确保 schema 兼容。
    同时启动单写线程队列（幂等，多次调用安全）。
    """
    if not os.path.exists(db_path):
        init_db(db_path)
    else:
        conn = sqlite3.connect(db_path)
        try:
            _enable_wal(conn)
            _ensure_reps_schema(conn)
            _ensure_analysis_logs_schema(conn)
            _ensure_sessions_schema(conn)
            _ensure_ml_schema(conn)
        finally:
            conn.close()
    _migrate_legacy_sessions(db_path)
    _start_write_worker(db_path)


def init_db(db_path=DB_PATH):
    """建表初始化（仅 CREATE TABLE IF NOT EXISTS，不删除现有数据）。"""
    conn = sqlite3.connect(db_path)
    _enable_wal(conn)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            rep_count INTEGER NOT NULL,
            v_mean REAL,
            rom REAL,
            rom_completion_pct REAL,
            left_knee_angle REAL,
            right_knee_angle REAL,
            trunk_angle REAL,
            dtw_similarity REAL,
            depth_offset_cm REAL,
            load_kg REAL,
            velocity_loss REAL,
            user_id INTEGER DEFAULT 1,
            set_number INTEGER,
            session_id TEXT,
            user_height REAL,
            pose_issues TEXT,
            calib_method TEXT,
            calib_is_fallback INTEGER,
            timing_source TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            user_name    TEXT NOT NULL DEFAULT 'qiqi',
            started_at   TEXT,
            ended_at     TEXT,
            source_type  TEXT,
            source_name  TEXT,
            notes        TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS standard_action (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_idx INTEGER NOT NULL,
            left_knee REAL,
            right_knee REAL,
            trunk REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            start_time TEXT NOT NULL,
            duration REAL,
            reps_count INTEGER DEFAULT 0,
            status TEXT NOT NULL,
            error_msg TEXT
        )
        """
    )
    _ensure_reps_schema(conn)
    _ensure_analysis_logs_schema(conn)
    _ensure_sessions_schema(conn)
    _ensure_ml_schema(conn)
    conn.commit()
    conn.close()


def insert_rep(
    db_path: str,
    rep_count: int,
    v_mean: float,
    rom: float,
    left_knee: Optional[float],
    right_knee: Optional[float],
    trunk: Optional[float],
    dtw_sim: Optional[float],
    depth_offset_cm: Optional[float] = None,
    load_kg: Optional[float] = None,
    velocity_loss: Optional[float] = None,
    user_name: str = "qiqi",
    rom_completion_pct: Optional[float] = None,
    set_number: Optional[int] = None,
    session_id: Optional[str] = None,
    user_height: Optional[float] = None,
    pose_issues: Optional[str] = None,
    calib_method: Optional[str] = None,
    calib_is_fallback: Optional[bool] = None,
    timing_source: Optional[str] = None,
) -> None:
    """将一条 rep 记录写入数据库。使用写队列（非阻塞），避免主线程锁竞争。"""
    uname = (user_name or "qiqi").strip() or "qiqi"
    uname = "_".join(uname.split())
    fallback_int = None if calib_is_fallback is None else (1 if calib_is_fallback else 0)
    _enqueue_write(
        """
        INSERT INTO reps (
            ts, rep_count, v_mean, rom, rom_completion_pct, left_knee_angle, right_knee_angle,
            trunk_angle, dtw_similarity, depth_offset_cm, load_kg, velocity_loss, user_name,
            set_number, session_id, user_height, pose_issues,
            calib_method, calib_is_fallback, timing_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            rep_count,
            v_mean,
            rom,
            rom_completion_pct,
            left_knee,
            right_knee,
            trunk,
            dtw_sim,
            depth_offset_cm,
            load_kg,
            velocity_loss,
            uname,
            set_number,
            session_id,
            user_height,
            pose_issues,
            calib_method,
            fallback_int,
            timing_source,
        ),
    )


def log_analysis_task(db_path, video_name, user_name: str = "qiqi", start_time=None, duration=None, reps_count=0, status="Success", error_msg=None):
    uname = (user_name or "qiqi").strip() or "qiqi"
    uname = "_".join(uname.split())
    # schema 已在 ensure_db_safe / init_db 启动阶段初始化，不需要重复调用
    _enqueue_write(
        """
        INSERT INTO analysis_logs (video_name, user_id, user_name, start_time, duration, reps_count, status, error_msg)
        VALUES (?, 1, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(video_name),
            uname,
            str(start_time),
            float(duration) if duration is not None else None,
            int(reps_count or 0),
            str(status),
            None if error_msg is None else str(error_msg),
        ),
    )


def get_recent_analysis_logs(db_path=DB_PATH, limit=10):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, video_name, user_id, start_time, duration, reps_count, status, error_msg
        FROM analysis_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_distinct_user_names(db_path: str = DB_PATH) -> list[str]:
    """获取数据库中所有出现过的 user_name（去重），用于多用户过滤。"""
    if not os.path.isfile(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA table_info(reps)")
        cols = {r[1] for r in cur.fetchall()}
        if "user_name" not in cols:
            conn.close()
            return ["qiqi"]
        cur.execute("SELECT DISTINCT user_name FROM reps WHERE user_name IS NOT NULL AND user_name != '' ORDER BY user_name")
        rows = cur.fetchall()
        return [str(r[0]) for r in rows if r[0]]
    except Exception:
        return ["qiqi"]
    finally:
        conn.close()


def get_all_sessions(db_path: str = DB_PATH) -> list:
    """
    获取所有训练 Session 列表，真正按 session_id 聚合（不再按日期）。
    返回: [{"id": session_id, "session_id": ..., "user_name": ...,
             "started_at": ..., "reps": count, "source_type": ...}, ...]
    """
    if not os.path.isfile(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        # 确保 sessions 表存在（兼容未迁移的库）
        _ensure_sessions_schema(conn)
        cur.execute(
            """
            SELECT s.session_id,
                   COALESCE(s.user_name, 'qiqi') AS user_name,
                   s.started_at,
                   s.ended_at,
                   s.source_type,
                   s.source_name,
                   COUNT(r.id) AS reps
            FROM sessions s
            LEFT JOIN reps r ON r.session_id = s.session_id
            GROUP BY s.session_id
            ORDER BY COALESCE(s.started_at, s.session_id) DESC
            """
        )
        rows = cur.fetchall()
        return [
            {
                "id":          str(r[0]),
                "session_id":  str(r[0]),
                "user_name":   str(r[1]),
                "started_at":  str(r[2]) if r[2] else "",
                "ended_at":    str(r[3]) if r[3] else "",
                "source_type": str(r[4]) if r[4] else "",
                "source_name": str(r[5]) if r[5] else "",
                "reps":        int(r[6]),
                # 向后兼容：旧代码用 "date" 字段
                "date":        str(r[2])[:10] if r[2] else str(r[0])[:10],
            }
            for r in rows
        ]
    finally:
        conn.close()


def delete_multiple_sessions(db_path: str, session_ids: list) -> int:
    """
    批量删除 Session：按真实 session_id 删除 reps 和 sessions 表中的记录。
    session_ids: session_id 字符串列表
    返回: 删除的 reps 行数
    """
    if not session_ids:
        return 0
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        placeholders = ",".join("?" * len(session_ids))
        cur.execute(
            f"DELETE FROM reps WHERE session_id IN ({placeholders})",
            session_ids,
        )
        deleted_reps = cur.rowcount
        cur.execute(
            f"DELETE FROM sessions WHERE session_id IN ({placeholders})",
            session_ids,
        )
        conn.commit()
        return deleted_reps
    finally:
        conn.close()


def update_set_load_kg(db_path: str, user_name: str, date_str: str, set_number: int, new_load_kg: float) -> int:
    """批量更新指定用户/日期/组号的 load_kg。返回受影响行数。"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE reps SET load_kg = ? WHERE user_name = ? AND substr(ts, 1, 10) = ? AND set_number = ?",
            (float(new_load_kg), str(user_name), str(date_str), int(set_number)),
        )
        affected = cur.rowcount
        conn.commit()
        return affected
    finally:
        conn.close()


def delete_set_reps(db_path: str, user_name: str, date_str: str, set_number: int) -> int:
    """精准删除指定用户/日期/组号的所有 Rep 记录。返回删除行数。"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM reps WHERE user_name = ? AND substr(ts, 1, 10) = ? AND set_number = ?",
            (str(user_name), str(date_str), int(set_number)),
        )
        deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def get_standard(db_path=DB_PATH):
    if not os.path.isfile(db_path):
        return None
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='standard_action'")
    if not cur.fetchone():
        conn.close()
        return None
    cur.execute("SELECT frame_idx, left_knee, right_knee, trunk FROM standard_action ORDER BY frame_idx")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return None
    return [(r[1], r[2], r[3]) for r in rows]


# ══════════════════════════════════════════════════════════════
# ML lifecycle persistence helpers
# ══════════════════════════════════════════════════════════════

def persist_set_summary(db_path: str, summary: "SetSummary") -> Optional[int]:
    """
    Write a SetSummary to the `sets` table.
    Returns the new row id, or None on failure.
    """
    from datetime import datetime as _dt
    try:
        conn = sqlite3.connect(db_path)
        _enable_wal(conn)
        started = _dt.fromtimestamp(summary.start_ts).isoformat() if summary.start_ts else None
        ended   = _dt.fromtimestamp(summary.end_ts).isoformat()   if summary.end_ts   else None
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO sets (
                session_id, user_name, set_number, mode, load_kg, reps,
                best_velocity, mean_velocity, velocity_loss_pct,
                trusted_rep_count, untrusted_rep_count, quality_reasons,
                started_at, ended_at, was_stopped_by_policy
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                summary.session_id, summary.user_name, summary.set_number,
                summary.mode, summary.load_kg, summary.reps,
                summary.best_velocity, summary.mean_velocity, summary.velocity_loss_pct,
                summary.trusted_rep_count, summary.untrusted_rep_count,
                ",".join(summary.quality_reasons),
                started, ended,
                1 if summary.was_stopped_by_policy else 0,
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        return row_id
    except Exception as e:
        logger.warning("persist_set_summary failed: %s", e)
        return None


# Import only when needed (avoid circular imports at module level)
def _get_set_summary_type():
    try:
        from vbt_training_modes import SetSummary
        return SetSummary
    except Exception:
        return None


def log_prediction(
    db_path: str,
    session_id: Optional[str],
    set_id: Optional[int],
    model_name: str,
    model_version: str,
    prediction_type: str,
    fatigue_risk: Optional[float],
    stop_probability: Optional[float],
    technique_anomaly: Optional[float],
    confidence: float,
    used_model: bool,
    recommendation: Optional[str] = None,
    recommendation_reason: Optional[str] = None,
) -> None:
    """Async-queue a prediction log entry."""
    _enqueue_write(
        """
        INSERT INTO prediction_logs (
            ts, session_id, set_id, model_name, model_version, prediction_type,
            fatigue_risk, stop_probability, technique_anomaly,
            confidence, used_model, recommendation, recommendation_reason
        ) VALUES (datetime('now'),?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            session_id, set_id, model_name, model_version, prediction_type,
            fatigue_risk, stop_probability, technique_anomaly,
            confidence, 1 if used_model else 0,
            recommendation, recommendation_reason,
        ),
    )


def log_user_feedback(
    db_path: str,
    session_id: Optional[str],
    set_id: Optional[int],
    user_name: str,
    rpe: Optional[float],
    felt_label: Optional[str],
    notes: Optional[str] = None,
) -> None:
    """Async-queue a user feedback entry."""
    _enqueue_write(
        """
        INSERT INTO user_feedback (ts, session_id, set_id, user_name, rpe, felt_label, notes)
        VALUES (datetime('now'),?,?,?,?,?,?)
        """,
        (session_id, set_id, user_name, rpe, felt_label, notes),
    )


def save_training_sample(
    db_path: str,
    session_id: Optional[str],
    set_id: Optional[int],
    user_name: str,
    features_json: str,
    label_fatigue_risk: Optional[float] = None,
    label_stop: Optional[int] = None,
    label_source: str = "auto",
    trust_score: float = 1.0,
    quality_reasons: Optional[str] = None,
) -> None:
    """Async-queue a training sample."""
    _enqueue_write(
        """
        INSERT INTO training_samples (
            session_id, set_id, user_name, created_at,
            features_json, label_fatigue_risk, label_stop,
            label_source, trust_score, quality_reasons
        ) VALUES (?,?,?,datetime('now'),?,?,?,?,?,?)
        """,
        (
            session_id, set_id, user_name,
            features_json,
            label_fatigue_risk, label_stop,
            label_source, trust_score, quality_reasons,
        ),
    )
