#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VBT 分析核心：几何计算、数据库、审计日志。"""

import os
import sqlite3
from datetime import datetime
from typing import Optional

import numpy as np

from physics_converter import REAL_HEIGHT_M, calculate_m_per_pixel, get_depth_offset, DEFAULT_SCALE

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
    cur.execute("PRAGMA table_info(analysis_logs)")
    cols = {r[1] for r in cur.fetchall()}
    if "user_name" not in cols:
        try:
            cur.execute("ALTER TABLE analysis_logs ADD COLUMN user_name TEXT")
            cur.execute("UPDATE analysis_logs SET user_name = 'qiqi' WHERE user_name IS NULL OR user_name = ''")
        except sqlite3.OperationalError:
            pass
    conn.commit()


def ensure_db_safe(db_path: str = DB_PATH) -> None:
    """断点续存：仅当 DB 文件不存在时建表；已存在则仅连接并确保 schema 兼容，严禁删除或重置数据。"""
    if not os.path.exists(db_path):
        init_db(db_path)
        return
    conn = sqlite3.connect(db_path)
    try:
        _ensure_reps_schema(conn)
        _ensure_analysis_logs_schema(conn)
    finally:
        conn.close()


def init_db(db_path=DB_PATH):
    """建表初始化（仅 CREATE TABLE IF NOT EXISTS，不删除现有数据）。"""
    conn = sqlite3.connect(db_path)
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
            pose_issues TEXT
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
) -> None:
    uname = (user_name or "qiqi").strip() or "qiqi"
    uname = "_".join(uname.split())
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS reps ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, rep_count INTEGER NOT NULL, "
        "v_mean REAL, rom REAL, rom_completion_pct REAL, left_knee_angle REAL, right_knee_angle REAL, trunk_angle REAL, "
        "dtw_similarity REAL, depth_offset_cm REAL, load_kg REAL, velocity_loss REAL, user_id INTEGER DEFAULT 1, "
        "user_name TEXT, set_number INTEGER, session_id TEXT, user_height REAL, pose_issues TEXT)"
    )
    _ensure_reps_schema(conn)
    cur.execute(
        """
        INSERT INTO reps (
            ts, rep_count, v_mean, rom, rom_completion_pct, left_knee_angle, right_knee_angle,
            trunk_angle, dtw_similarity, depth_offset_cm, load_kg, velocity_loss, user_name,
            set_number, session_id, user_height, pose_issues
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ),
    )
    conn.commit()
    conn.close()


def log_analysis_task(db_path, video_name, user_name: str = "qiqi", start_time=None, duration=None, reps_count=0, status="Success", error_msg=None):
    uname = (user_name or "qiqi").strip() or "qiqi"
    uname = "_".join(uname.split())
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    _ensure_analysis_logs_schema(conn)
    cur.execute(
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
    conn.commit()
    conn.close()


def get_recent_analysis_logs(db_path=DB_PATH, limit=10):
    init_db(db_path)
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


def get_all_sessions(db_path: str = DB_PATH) -> list[dict]:
    """
    获取所有训练 Session 列表。Session 按日期分组（同一天的 Reps 为一组）。
    返回: [{"id": date_str, "date": date_str, "reps": count}, ...]
    """
    if not os.path.isfile(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT substr(ts, 1, 10) AS date, COUNT(*) AS reps
            FROM reps
            GROUP BY substr(ts, 1, 10)
            ORDER BY date DESC
            """
        )
        rows = cur.fetchall()
        return [
            {"id": str(r[0]), "date": str(r[0]), "reps": int(r[1])}
            for r in rows
        ]
    finally:
        conn.close()


def delete_multiple_sessions(db_path: str, session_ids: list[str]) -> int:
    """
    批量删除 Session（按日期）。删除该日期下所有 Reps 记录。
    session_ids: 日期字符串列表，如 ["2025-03-09", "2025-03-08"]
    返回: 删除的行数
    """
    if not session_ids:
        return 0
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        placeholders = ",".join("?" * len(session_ids))
        cur.execute(
            f"DELETE FROM reps WHERE substr(ts, 1, 10) IN ({placeholders})",
            session_ids,
        )
        deleted = cur.rowcount
        conn.commit()
        return deleted
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
