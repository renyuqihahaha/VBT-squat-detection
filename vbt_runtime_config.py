#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VBT 运行时共享配置（JSON 持久化）。"""

import json
import os
from typing import Optional

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vbt_config.json")
DEFAULT_USER_NAME = "qiqi"
DEFAULT_LOAD_KG = 60.0
DEFAULT_USER_HEIGHT_CM = 175.0


def _sanitize_user_name(raw: Optional[str]) -> str:
    """清洗用户姓名：去首尾空格，中间空格替换为下划线，确保文件系统兼容。"""
    if raw is None or not isinstance(raw, str):
        return DEFAULT_USER_NAME
    s = str(raw).strip()
    if not s:
        return DEFAULT_USER_NAME
    return "_".join(s.split())


def _default_config():
    return {
        "active_user_name": DEFAULT_USER_NAME,
        "current_load_kg": DEFAULT_LOAD_KG,
        "user_height_cm": DEFAULT_USER_HEIGHT_CM,
    }


def _load():
    if not os.path.isfile(CONFIG_PATH):
        return _default_config()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _default_config()
        # 兼容旧版 active_user_id：若存在则映射为默认姓名
        user_name = data.get("active_user_name")
        if user_name is None and "active_user_id" in data:
            user_name = DEFAULT_USER_NAME
        return {
            "active_user_name": _sanitize_user_name(user_name or DEFAULT_USER_NAME),
            "current_load_kg": float(data.get("current_load_kg", DEFAULT_LOAD_KG)),
            "user_height_cm": float(data.get("user_height_cm", DEFAULT_USER_HEIGHT_CM)),
        }
    except Exception:
        return _default_config()


def save_runtime_config(load_kg, user_name=None, user_height_cm=None):
    name = _sanitize_user_name(user_name) if user_name is not None else get_current_user_name()
    payload = {
        "active_user_name": name,
        "current_load_kg": float(load_kg),
        "user_height_cm": float(user_height_cm) if user_height_cm is not None else DEFAULT_USER_HEIGHT_CM,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return CONFIG_PATH


def get_current_user_name() -> str:
    cfg = _load()
    return _sanitize_user_name(cfg.get("active_user_name", DEFAULT_USER_NAME))


def sanitize_user_name(raw: Optional[str]) -> str:
    """清洗用户姓名：去首尾空格，中间空格替换为下划线，确保文件系统兼容。"""
    return _sanitize_user_name(raw)


def get_current_load_kg():
    cfg = _load()
    try:
        return float(cfg.get("current_load_kg", DEFAULT_LOAD_KG))
    except (TypeError, ValueError):
        return DEFAULT_LOAD_KG


def get_user_height_cm():
    cfg = _load()
    try:
        v = float(cfg.get("user_height_cm", DEFAULT_USER_HEIGHT_CM))
        return v if 100 <= v <= 250 else DEFAULT_USER_HEIGHT_CM
    except (TypeError, ValueError):
        return DEFAULT_USER_HEIGHT_CM

