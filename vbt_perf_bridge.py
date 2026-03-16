#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBT 性能监控数据桥：视觉引擎与 Dashboard 之间的轻量级共享缓存。
使用 JSON 文件原子写入，防止读写冲突。
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Optional

PERF_FILE = os.path.join(tempfile.gettempdir(), "vbt_performance.json")
STALE_SECONDS = 2.0


def write_stats(fps: float, latency_ms: float) -> None:
    """
    将 FPS 和 Latency 写入共享缓存。
    使用原子替换（先写临时文件再 rename），防止读取冲突。
    """
    try:
        data = {
            "fps": float(fps),
            "latency_ms": float(latency_ms),
            "ts": time.time(),
        }
        tmp = PERF_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, PERF_FILE)
    except Exception:
        pass


def read_stats() -> Optional[dict]:
    """
    从共享缓存读取性能指标。
    若文件不存在或超过 STALE_SECONDS 未更新，返回 None。
    """
    try:
        if not os.path.exists(PERF_FILE):
            return None
        with open(PERF_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("ts", 0)
        if time.time() - ts > STALE_SECONDS:
            return None
        return data
    except Exception:
        return None
