#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBT Command Center：本地视频分析 + 实时采集 + 数据看板。
"""

from __future__ import annotations

import logging
import os
import sqlite3
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from vbt_analytics_pro import DB_PATH, ensure_db_safe, get_all_sessions, delete_multiple_sessions, get_distinct_user_names, update_set_load_kg, delete_set_reps

# 杠铃片物理映射（单边挂片，直径用于视觉标定）
PLATE_SPECS = {
    25: {"color": "#FF3B30", "diameter_cm": 45.0},
    20: {"color": "#007AFF", "diameter_cm": 45.0},
    15: {"color": "#FFCC00", "diameter_cm": 45.0},
    10: {"color": "#34C759", "diameter_cm": 32.0},
    5: {"color": "#FFFFFF", "diameter_cm": 23.0},
}
PLATE_VISUALS = {
    25: {"color": "#FF3B30", "height": "80px", "width": "18px"},
    20: {"color": "#007AFF", "height": "80px", "width": "15px"},
    15: {"color": "#FFCC00", "height": "80px", "width": "12px"},
    10: {"color": "#34C759", "height": "55px", "width": "15px"},
    5: {"color": "#E5E5EA", "height": "40px", "width": "12px"},
}
BAR_WEIGHT = 20.0
MAX_PLATES_PER_SIDE = 8
from vbt_ai_advisor import (
    get_unified_lvp_prediction,
    assess_daily_readiness,
    get_training_advice,
    get_recommended_load,
    predict_fatigue,
    assess_set_fatigue,
    predict_1rm_time_weighted,
    LVPModel,
    LVP1RMPrediction,
    FatiguePrediction,
    SetFatigueAssessment,
    MIN_DATA_POINTS,
    MVT_M_S,
)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def _update_perf_placeholders(stats: dict, placeholders: dict, perf_state: dict) -> None:
    """循环内同步更新侧边栏性能占位符（EMA 平滑 + 每 10 帧降频刷新）。"""
    if not placeholders:
        return
    perf_state["frame_count"] += 1
    alpha = perf_state["alpha"]

    current_cpu = float(psutil.cpu_percent(interval=None)) if HAS_PSUTIL else -1.0
    current_fps = float(stats.get("fps") or 0)
    current_lat = float(stats.get("latency_ms") or 0)

    perf_state["ema_cpu"] = current_cpu if perf_state["ema_cpu"] is None else (alpha * current_cpu) + ((1 - alpha) * perf_state["ema_cpu"])
    perf_state["ema_fps"] = current_fps if perf_state["ema_fps"] is None else (alpha * current_fps) + ((1 - alpha) * perf_state["ema_fps"])
    perf_state["ema_lat"] = current_lat if perf_state["ema_lat"] is None else (alpha * current_lat) + ((1 - alpha) * perf_state["ema_lat"])

    if perf_state["frame_count"] % 10 == 0:
        try:
            cpu_str = f"{perf_state['ema_cpu']:.1f} %" if perf_state["ema_cpu"] >= 0 else "N/A"
            mem_pct = float(psutil.virtual_memory().percent) if HAS_PSUTIL else -1.0
            mem_str = f"{mem_pct:.1f} %" if mem_pct >= 0 else "N/A"
            placeholders["cpu"].metric("💻 CPU", cpu_str)
            placeholders["ram"].metric("🧠 RAM", mem_str)
            placeholders["fps"].metric("🎞️ FPS", f"{perf_state['ema_fps']:.1f}")
            placeholders["latency"].metric("⏱️ Latency", f"{perf_state['ema_lat']:.0f} ms")
        except Exception:
            pass


def _get_hw_metrics_cached() -> tuple[float, float]:
    """CPU/RAM 每 2 秒刷新一次；CPU 使用 interval=0.1 确保即时波动感。"""
    cache = st.session_state.get("_hw_cache", {})
    now = time.time()
    if cache and (now - cache.get("ts", 0)) < 2.0:
        return cache.get("cpu", -1.0), cache.get("mem", -1.0)
    if not HAS_PSUTIL:
        return -1.0, -1.0
    try:
        cpu = float(psutil.cpu_percent(interval=0.1))
        mem = float(psutil.virtual_memory().percent)
        st.session_state["_hw_cache"] = {"cpu": cpu, "mem": mem, "ts": now}
        return cpu, mem
    except Exception:
        return -1.0, -1.0

logger = logging.getLogger("vbt_dashboard")


def _load_to_plates(total_kg: float) -> list[int]:
    """将总负重解析为单边杠铃片列表（贪心，优先大片）。"""
    if total_kg <= BAR_WEIGHT:
        return []
    per_side = (total_kg - BAR_WEIGHT) / 2
    out = []
    for kg in sorted(PLATE_SPECS.keys(), reverse=True):
        while per_side >= kg:
            out.append(kg)
            per_side -= kg
    return out


def _query_df(sql: str, params: tuple = (), db_path: Optional[str] = None) -> pd.DataFrame:
    """执行 SQL 并返回 DataFrame。"""
    db = db_path or DB_PATH
    conn = sqlite3.connect(db)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def _apply_dark_theme() -> None:
    """应用深色主题样式。"""
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", sans-serif;
        }
        .stMetric {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #374151;
        }
        .fatigue-warn {
            color: #f97316 !important;
            font-weight: 700;
        }
        .fatigue-danger {
            color: #ef4444 !important;
            font-weight: 700;
        }
        .rom-low {
            color: #eab308 !important;
            font-weight: 600;
        }
        .rom-ok {
            color: #22c55e !important;
            font-weight: 600;
        }
        .nowrap { white-space: nowrap; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_dual_mode_sidebar() -> tuple[str, int, float]:
    """侧边栏：双模路由与运行配置。"""
    st.sidebar.subheader("工作模式")
    mode = st.sidebar.radio(
        "选择数据源",
        ["本地视频分析", "实时采集 (树莓派)"],
        index=0,
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("运行配置")
    from vbt_runtime_config import get_current_user_name, get_current_load_kg, get_user_height_cm, save_runtime_config, sanitize_user_name
    raw_name = st.sidebar.text_input(
        "用户姓名 (English Name)",
        value=get_current_user_name(),
        key="sidebar_user_name",
        help="自动去除首尾空格，中间空格替换为下划线",
    )
    user_name = sanitize_user_name(raw_name) if raw_name else get_current_user_name()
    if "user_name" not in st.session_state:
        st.session_state["user_name"] = user_name
    st.session_state["user_name"] = user_name

    if "current_plates" not in st.session_state:
        init_load = float(get_current_load_kg())
        st.session_state["current_plates"] = _load_to_plates(init_load) if init_load > BAR_WEIGHT else []

    is_bodyweight = st.sidebar.checkbox("🏃 自重/徒手深蹲 (无杠铃)", value=st.session_state.get("is_bodyweight", False), key="sidebar_bodyweight")
    st.session_state["is_bodyweight"] = is_bodyweight

    if is_bodyweight:
        st.sidebar.info("已切换至自重模式：系统将锁定站立姿态进行视觉标定，并追踪您的髋部重心轨迹。")
        total_load = 0.0
        max_diameter = 0.0
        st.session_state["_plate_max_diameter_cm"] = 0.0
    else:
        st.sidebar.subheader("🏋️ 负重设置 (交互式组装)")
        plate_cols = st.sidebar.columns(6)
        for i, kg in enumerate(sorted(PLATE_SPECS.keys())):
            with plate_cols[i]:
                if st.button(f"+{kg}", key=f"plate_{kg}", use_container_width=True):
                    plates = st.session_state["current_plates"]
                    if len(plates) < MAX_PLATES_PER_SIDE:
                        plates = list(plates) + [kg]
                        st.session_state["current_plates"] = plates
                        st.rerun()
        with plate_cols[5]:
            if st.button("清空", key="plate_clear", use_container_width=True):
                st.session_state["current_plates"] = []
                st.rerun()

        if st.session_state["current_plates"]:
            html_str = textwrap.dedent("""
            <div style="display: flex; align-items: center; justify-content: flex-start; padding: 10px 0 20px 0; overflow-x: auto;">
            <div style="width: 20px; height: 16px; background-color: #8E8E93; border-radius: 2px 0 0 2px;"></div>
            <div style="width: 10px; height: 24px; background-color: #636366; margin-right: 2px;"></div>
            """)
            for plate in st.session_state["current_plates"]:
                vis = PLATE_VISUALS.get(plate, {"color": "#333", "height": "40px", "width": "10px"})
                html_str += f'<div style="width: {vis["width"]}; height: {vis["height"]}; background-color: {vis["color"]}; border-radius: 3px; margin-right: 2px; box-shadow: 1px 0px 3px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1);"></div>\n'
            html_str += textwrap.dedent("""
            <div style="flex-grow: 1; height: 16px; background-color: #8E8E93; border-radius: 0 2px 2px 0; min-width: 50px;"></div>
            </div>
            """)
            st.sidebar.markdown(html_str, unsafe_allow_html=True)

        plates = st.session_state["current_plates"]
        if plates:
            st.sidebar.caption(f"当前挂片: {' + '.join(str(p) for p in plates)} kg/边")
        total_load = BAR_WEIGHT + sum(plates) * 2
        max_diameter = max(PLATE_SPECS[p]["diameter_cm"] for p in plates) if plates else 0.0
        st.session_state["_plate_max_diameter_cm"] = max_diameter

    plates = [] if is_bodyweight else st.session_state["current_plates"]

    m1, m2 = st.sidebar.columns(2)
    with m1:
        st.metric("总负重", f"{total_load:.1f} kg")
    with m2:
        st.metric("标定直径", f"{max_diameter:.0f} cm" if max_diameter else "—")

    load_kg = total_load

    user_height = st.sidebar.number_input(
        "身高 (cm)", min_value=100.0, max_value=250.0,
        value=float(get_user_height_cm()), step=1.0,
        help="用于像素-米动态校准，直接影响速度精度",
    )
    if st.sidebar.button("保存配置", use_container_width=True, key="save_full_config"):
        save_runtime_config(load_kg=load_kg, user_name=user_name, user_height_cm=user_height)
        st.session_state["user_name"] = user_name
        st.sidebar.success("配置已保存")

    st.sidebar.markdown("---")
    st.sidebar.subheader("当前训练配置")
    default_session = datetime.now().strftime("%Y-%m-%d")
    if "current_session_id" not in st.session_state:
        st.session_state["current_session_id"] = default_session
    if "current_set_number" not in st.session_state:
        st.session_state["current_set_number"] = 1

    session_id = st.sidebar.text_input(
        "课号 (Session ID)", value=st.session_state["current_session_id"], key="sidebar_session_id",
    )
    st.session_state["current_session_id"] = session_id

    set_number = st.sidebar.number_input(
        "组序号 (Set #)", min_value=1,
        value=int(st.session_state["current_set_number"]),
        step=1, key="sidebar_set_number",
    )
    st.session_state["current_set_number"] = int(set_number)

    st.sidebar.caption(f"📋 课号: **{session_id}** | 第 **{set_number}** 组 | 负重: **{load_kg}** kg")

    st.sidebar.markdown("---")
    st.sidebar.subheader("采集选项")
    if "rt_mirror" not in st.session_state:
        st.session_state["rt_mirror"] = False
    st.sidebar.checkbox("镜像翻转 (实时采集)", value=st.session_state.get("rt_mirror", False), key="sidebar_mirror")
    st.session_state["rt_mirror"] = st.session_state.get("sidebar_mirror", False)

    if "pose_diag_off" not in st.session_state:
        st.session_state["pose_diag_off"] = False
    st.sidebar.checkbox("静默模式 (关闭姿态诊断)", value=st.session_state.get("pose_diag_off", False), key="sidebar_pose_off")
    st.session_state["pose_diag_off"] = st.session_state.get("sidebar_pose_off", False)

    if "use_plate_calibration" not in st.session_state:
        st.session_state["use_plate_calibration"] = False
    st.sidebar.checkbox(
        "使用标准杠铃片标定 (45cm)",
        value=st.session_state.get("use_plate_calibration", False),
        key="sidebar_plate_calib",
        help="画面中需可见标准杠铃片，系统将自动检测其像素直径进行标定",
    )
    st.session_state["use_plate_calibration"] = st.session_state.get("sidebar_plate_calib", False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("AI 训练模式")
    training_goal_ui = st.sidebar.selectbox(
        "训练目标（影响疲劳阈值）",
        options=["Strength", "Hypertrophy", "Power"],
        index=0,
        format_func=lambda x: {"Strength": "最大力量 (Strength)", "Hypertrophy": "肌肥大 (Hypertrophy)", "Power": "爆发力 (Power)"}[x],
        key="training_goal_select",
    )
    training_mode = "hypertrophy" if training_goal_ui == "Hypertrophy" else "strength"
    thresh_pct = "15%" if training_goal_ui == "Power" else "20%" if training_goal_ui == "Strength" else "35%"
    st.sidebar.caption(f"当前目标: **{training_goal_ui}** | 流失阈值: **{thresh_pct}**")

    # ── 边缘计算引擎状态（唯一实例，st.empty 占位符，循环内同步更新）──
    st.sidebar.divider()
    st.sidebar.subheader("⚡ 边缘计算引擎状态")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        cpu_placeholder = st.empty()
        latency_placeholder = st.empty()
    with col2:
        ram_placeholder = st.empty()
        fps_placeholder = st.empty()

    cpu_placeholder.metric("💻 CPU", "待机...")
    ram_placeholder.metric("🧠 RAM", "待机...")
    latency_placeholder.metric("⏱️ Latency", "待机...")
    fps_placeholder.metric("🎞️ FPS", "待机...")

    st.session_state["_perf_placeholders"] = {
        "cpu": cpu_placeholder,
        "ram": ram_placeholder,
        "latency": latency_placeholder,
        "fps": fps_placeholder,
    }

    return mode, user_name, load_kg


def _render_mode_a_realtime() -> None:
    """模式 A：实时采集 — 一键启动摄像头，实时 VBT 分析，与终端版视觉对齐。"""
    import threading
    import traceback

    st.subheader("实时采集 (树莓派)")

    for key, default in [("rt_running", False), ("rt_stop_flag", None), ("rt_camera_idx", 0)]:
        if key not in st.session_state:
            st.session_state[key] = default

    is_running = st.session_state["rt_running"]

    col_cfg, col_ctrl = st.columns([1, 1])
    with col_cfg:
        camera_index = st.number_input(
            "摄像头索引", min_value=0, value=int(st.session_state["rt_camera_idx"]),
            step=1, disabled=is_running,
            help="树莓派 CSI/USB 摄像头索引，常见: 0, 2, 4, 8",
        )
        if not is_running:
            st.session_state["rt_camera_idx"] = int(camera_index)
        record_on = st.checkbox("录制原始视频", value=True, disabled=is_running, key="rt_record_cb")

    with col_ctrl:
        if not is_running:
            if st.button("▶ 开始采集", type="primary", use_container_width=True, key="rt_start"):
                st.session_state["rt_running"] = True
                st.session_state["rt_stop_flag"] = threading.Event()
                st.rerun()
        else:
            if st.button("⏹ 停止采集", type="secondary", use_container_width=True, key="rt_stop"):
                flag = st.session_state.get("rt_stop_flag")
                if flag is not None:
                    flag.set()
                st.session_state["rt_running"] = False
                st.rerun()

    if not is_running:
        st.info("点击「开始采集」启动摄像头，系统将自动进行实时深蹲分析。")
        return

    try:
        from vbt_cv_engine import process_squat_video
        import cv2
    except ImportError as e:
        st.error(f"无法加载 CV 引擎: {e}")
        st.session_state["rt_running"] = False
        return

    from vbt_runtime_config import get_user_height_cm

    rt_set = int(st.session_state.get("current_set_number", 1))
    rt_session = str(st.session_state.get("current_session_id", datetime.now().strftime("%Y-%m-%d")))
    rt_height = float(get_user_height_cm())
    stop_flag = st.session_state["rt_stop_flag"]
    cam_idx = int(st.session_state["rt_camera_idx"])

    st.caption(f"📋 课号: {rt_session} | 第 {rt_set} 组 | 身高: {rt_height:.0f}cm | 摄像头: {cam_idx}")

    status_slot = st.empty()
    status_slot.info(f"正在初始化摄像头 (索引 {cam_idx})，请稍候...")

    video_slot = st.empty()
    rec_info_slot = st.empty()
    calib_slot = st.empty()
    col_left, col_right = st.columns([1, 1])
    with col_left:
        metrics_slot = st.empty()
    with col_right:
        chart_slot = st.empty()
    summary_slot = st.empty()

    rep_velocities: list[float] = []
    rep_rom_completions: list[float] = []
    last_rep_count = 0
    calib_warned = False
    rec_notified = False
    gen = None
    pose_diag_on = not st.session_state.get("pose_diag_off", False)

    pose_slot = st.empty()
    fatigue_slot = st.empty()

    try:
        plate_diam = st.session_state.get("_plate_max_diameter_cm") or 45.0
        use_plate = st.session_state.get("use_plate_calibration", False)
        is_bw = st.session_state.get("is_bodyweight", False)
        gen = process_squat_video(
            cam_idx, stop_flag=stop_flag,
            set_number=rt_set, session_id=rt_session,
            user_height_cm=rt_height,
            record_video=record_on,
            pose_diag_enabled=pose_diag_on,
            use_plate_calibration=use_plate and not is_bw,
            plate_diameter_cm=plate_diam if (use_plate and not is_bw and plate_diam > 0) else None,
            is_bodyweight=is_bw,
        )
        if gen is None:
            status_slot.error(
                f"无法打开摄像头 (索引 {cam_idx})。\n\n"
                "请检查: 1) USB 线是否插紧  2) 其他程序是否占用  3) 尝试其他索引 (0, 2, 4, 8)"
            )
            st.session_state["rt_running"] = False
            return

        status_slot.success(f"摄像头 {cam_idx} 已就绪，开始实时分析...")
        time.sleep(0.3)
        status_slot.empty()

        perf_state = {"ema_cpu": None, "ema_fps": None, "ema_lat": None, "frame_count": 0, "alpha": 0.2}
        placeholders = st.session_state.get("_perf_placeholders", {})

        for frame_bgr, stats in gen:
            if stop_flag and stop_flag.is_set():
                break
            _update_perf_placeholders(stats, placeholders, perf_state)

            if st.session_state.get("rt_mirror", False):
                frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            video_slot.image(frame_rgb, channels="RGB", use_container_width=True)

            if not rec_notified and stats.get("recording_path"):
                rec_file = os.path.basename(stats["recording_path"])
                rec_info_slot.info(f"🔴 正在录制视频至: recordings/{rec_file}")
                rec_notified = True

            if not calib_warned and stats.get("calibration_fallback"):
                calib_slot.warning("⚠️ 未检测到踝关节，已回退至躯干校准。建议拍摄全身以获得更准的速度数据。")
                calib_warned = True

            rc = stats.get("reps", 0)
            if rc > last_rep_count and rc > 0:
                rep_velocities.append(stats.get("current_vel", 0.0))
                rom_v = stats.get("rom_completion_pct")
                if rom_v is not None:
                    rep_rom_completions.append(float(rom_v))
                last_rep_count = rc

            with metrics_slot.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Reps", stats.get("reps", 0))
                cur_mean = rep_velocities[-1] if rep_velocities else stats.get("current_vel", 0.0)
                best_mean = max(rep_velocities) if rep_velocities else stats.get("best_vel", 0.0)
                m2.metric("末次均速", f"{cur_mean:.3f} m/s")
                m3.metric("最佳均速", f"{best_mean:.3f} m/s")

                m4, m5, m6 = st.columns(3)
                v_loss = ((best_mean - cur_mean) / best_mean * 100) if (rep_velocities and best_mean > 0) else stats.get("velocity_loss_pct", 0.0)
                m4.metric("速度损失", f"{v_loss:.1f}%")
                m5.metric("FPS", f"{stats.get('fps', 0):.0f}")

                rom_pct = stats.get("rom_completion_pct")
                rom_val = float(rom_pct) if rom_pct is not None else 0.0
                rom_str = f"{rom_val:.1f}%"
                if rom_val >= 90:
                    m6.markdown(f'<span class="rom-ok">完成度: {rom_str} ✓</span>', unsafe_allow_html=True)
                elif 0 < rom_val < 85:
                    m6.markdown(f'<span class="rom-low">完成度: {rom_str} ⚠</span>', unsafe_allow_html=True)
                else:
                    m6.metric("完成度", rom_str if rom_val > 0 else "—")

            pd_info = stats.get("pose_diag")
            if pd_info and pose_diag_on:
                with pose_slot.container():
                    ps1, ps2, ps3, ps4 = st.columns(4)
                    score = pd_info.get("score", 100)
                    score_color = "normal" if score >= 80 else ("off" if score >= 60 else "inverse")
                    ps1.metric("技术评分", f"{score:.0f}", delta=None)
                    knee_r = pd_info.get("knee_ratio", 1.0)
                    ps2.metric("膝踝比", f"{knee_r:.2f}", delta="内扣" if pd_info.get("knee_valgus") else "正常",
                               delta_color="inverse" if pd_info.get("knee_valgus") else "normal")
                    torso_a = pd_info.get("torso_angle", 0)
                    ps3.metric("躯干倾角", f"{torso_a:.1f}°", delta="代偿" if pd_info.get("torso_lean_warning") else "正常",
                               delta_color="inverse" if pd_info.get("torso_lean_warning") else "normal")
                    drift_r = pd_info.get("hip_x_drift_ratio", 0)
                    ps4.metric("重心偏移", f"{drift_r:.2%}", delta="不稳" if pd_info.get("stability_warning") else "稳定",
                               delta_color="inverse" if pd_info.get("stability_warning") else "normal")

            fat_pred = None
            best_v = max(rep_velocities) if rep_velocities else stats.get("best_vel", 0)
            if len(rep_velocities) >= 3 and best_v > 0:
                fat_pred = predict_fatigue(rep_velocities, best_v)
            training_goal = st.session_state.get("training_goal_select", "Strength")
            fat_assess = assess_set_fatigue(rep_velocities, training_goal) if rep_velocities else None
            if fat_assess or fat_pred:
                with fatigue_slot.container():
                    if fat_assess:
                        status_colors = {"STOP": "#ef4444", "Fatigued": "#f97316", "Warning": "#eab308", "Optimal": "#22c55e"}
                        color = status_colors.get(fat_assess.status, "#6b7280")
                        st.markdown(
                            f'<div style="padding:0.5rem 1rem; border-radius:8px; background:{color}22; border-left:4px solid {color}; font-weight:600;">'
                            f'{fat_assess.message} <span style="font-size:0.85rem; opacity:0.9;">(流失 {fat_assess.velocity_loss_pct:.1f}% / 阈值 {fat_assess.threshold_pct:.0f}%)</span></div>',
                            unsafe_allow_html=True,
                        )
                    if fat_pred:
                        fc1, fc2, fc3 = st.columns(3)
                        fc1.metric("预测下把速度", f"{fat_pred.predicted_next_v:.3f} m/s")
                        fc2.metric("预测衰减", f"{fat_pred.predicted_loss_pct:.1f}%")
                        fc3.metric("置信度", f"{fat_pred.confidence:.0%}")
                        if fat_pred.failure_warning and not (fat_assess and fat_assess.status == "STOP"):
                            st.error(f"⚠️ {fat_pred.message}")
                        elif not fat_assess or fat_assess.status == "Optimal":
                            st.caption(fat_pred.message)

            if len(rep_velocities) >= 2 and HAS_PLOTLY:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(rep_velocities) + 1)),
                    y=rep_velocities, mode="lines+markers",
                    name="向心速度", line=dict(color="#22c55e", width=2), marker=dict(size=8),
                ))
                baseline_vel = max(rep_velocities)
                thresh = 0.20 if training_goal == "Strength" else (0.35 if training_goal == "Hypertrophy" else 0.15)
                if baseline_vel > 0:
                    cutoff_vel = baseline_vel * (1 - thresh)
                    fig.add_hline(y=cutoff_vel, line_dash="dash", line_color="#ef4444",
                                  annotation_text=f"{thresh*100:.0f}% 流失线")
                if fat_pred and fat_pred.predicted_velocities:
                    pred_x = list(range(1, len(fat_pred.predicted_velocities) + 1))
                    fig.add_trace(go.Scatter(
                        x=pred_x, y=fat_pred.predicted_velocities,
                        mode="lines", name="AI 预测",
                        line=dict(color="#f59e0b", width=2, dash="dash"),
                    ))
                fig.update_layout(
                    title="速度衰减曲线", xaxis_title="Rep", yaxis_title="m/s",
                    template="plotly_dark", height=280, margin=dict(t=40, b=30),
                )
                chart_slot.plotly_chart(fig, use_container_width=True, key=f"rt_chart_{time.time_ns()}")

            time.sleep(0.01)

    except Exception as e:
        if "Rerun" not in type(e).__name__ and "Stop" not in type(e).__name__:
            st.error(f"采集异常: {e}")
            st.code(traceback.format_exc(), language="text")
    finally:
        if gen is not None:
            try:
                gen.close()
            except Exception:
                pass
        st.session_state["rt_running"] = False
        st.session_state["rt_stop_flag"] = None
        logger.info("实时采集已停止，资源已释放")

    if rep_velocities:
        with summary_slot.container():
            st.markdown("---")
            st.subheader("本组采集总结")
            rt_goal = "Hypertrophy" if st.session_state.get("ai_mode_toggle", False) else "Strength"
            rt_assess = assess_set_fatigue(rep_velocities, rt_goal)
            if rt_assess.status == "STOP":
                st.error(f"🚨 {rt_assess.message}")
            elif rt_assess.status == "Fatigued":
                st.warning(f"⚠️ {rt_assess.message} (流失 {rt_assess.velocity_loss_pct:.1f}%)")
            elif rt_assess.status == "Warning":
                st.warning(f"⚡ {rt_assess.message}")
            else:
                st.success(f"✅ {rt_assess.message}")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("总 Reps", len(rep_velocities))
            sc2.metric("最佳均速", f"{max(rep_velocities):.3f} m/s")
            avg_rom = sum(rep_rom_completions) / len(rep_rom_completions) if rep_rom_completions else 0.0
            sc3.metric("平均 ROM", f"{avg_rom:.1f}%")

            next_set = rt_set + 1
            st.info(f"第 {rt_set} 组采集完成！准备好第 {next_set} 组了吗？")
            if st.button(f"▶ 开始第 {next_set} 组", type="primary", key="rt_next_set"):
                st.session_state["current_set_number"] = next_set
                st.rerun()


def _list_local_videos() -> list[tuple[str, str]]:
    """扫描项目目录下的视频文件，返回 (显示名, 绝对路径)。"""
    base = Path(__file__).resolve().parent
    exts = (".mp4", ".mov", ".avi")
    out: list[tuple[str, str]] = []
    for d in [base, base / "videos", base.parent]:
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in exts:
                out.append((f.name, str(f.resolve())))
    return out


def _render_mode_b_upload() -> None:
    """模式 B：本地视频分析 — 纯内存流分析，预建占位符，同步循环渲染。"""
    import tempfile
    import traceback

    st.subheader("本地视频分析")
    st.caption("选择或上传视频，点击「开始分析」后实时展示骨架追踪与速度曲线（纯内存分析，不保存文件）")

    source_tab = st.radio("视频来源", ["从项目目录选择", "上传文件"], horizontal=True, key="video_source")

    video_path: Optional[str] = None
    temp_path_to_clean: Optional[str] = None

    if source_tab == "从项目目录选择":
        local_videos = _list_local_videos()
        if not local_videos:
            st.info("项目目录下未找到 .mp4 / .mov / .avi 文件，请使用「上传文件」或放入 videos/ 目录。")
        else:
            options = ["(请选择)"] + [name for name, _ in local_videos]
            selected = st.selectbox("选择视频", options=options, key="local_video_select")
            if selected != "(请选择)":
                for name, path in local_videos:
                    if name == selected:
                        video_path = path
                        break
    else:
        uploaded = st.file_uploader("上传深蹲视频", type=["mp4", "mov"], help="支持 .mp4 / .mov", key="upload_video")
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as f:
                f.write(uploaded.getvalue())
                temp_path_to_clean = f.name
                video_path = temp_path_to_clean

    start_btn = st.button("开始分析", type="primary", use_container_width=True, key="start_analysis_btn")

    if not start_btn or video_path is None:
        if start_btn and video_path is None:
            st.warning("请先选择或上传视频文件。")
        return

    if not os.path.exists(video_path):
        st.error(f"视频文件不存在: {video_path}")
        return

    try:
        from vbt_cv_engine import process_squat_video
        import cv2
    except ImportError as e:
        st.error(f"无法加载 CV 引擎: {e}")
        return

    col_left, col_right = st.columns([1, 1])
    with col_left:
        video_container = st.empty()
    with col_right:
        chart_container = st.empty()
        metrics_container = st.empty()
    summary_container = st.empty()

    rep_velocities: list[float] = []
    rep_rom_completions: list[float] = []
    last_rep_count = 0
    final_stats: Optional[dict] = None
    last_frame_rgb = None

    cur_set = int(st.session_state.get("current_set_number", 1))
    cur_session = str(st.session_state.get("current_session_id", datetime.now().strftime("%Y-%m-%d")))

    from vbt_runtime_config import get_user_height_cm
    cur_height = float(get_user_height_cm())
    st.caption(f"📋 课号: {cur_session} | 第 {cur_set} 组 | 身高: {cur_height:.0f}cm")

    calib_warning_shown = False
    calib_placeholder = st.empty()

    pose_diag_on_b = not st.session_state.get("pose_diag_off", False)
    upload_pose_slot = st.empty()
    upload_fatigue_slot = st.empty()

    try:
        plate_diam = st.session_state.get("_plate_max_diameter_cm") or 45.0
        use_plate = st.session_state.get("use_plate_calibration", False)
        is_bw = st.session_state.get("is_bodyweight", False)
        gen = process_squat_video(
            video_path, set_number=cur_set, session_id=cur_session,
            user_height_cm=cur_height,
            pose_diag_enabled=pose_diag_on_b,
            use_plate_calibration=use_plate and not is_bw,
            plate_diameter_cm=plate_diam if (use_plate and not is_bw and plate_diam > 0) else None,
            is_bodyweight=is_bw,
        )
        if gen is None:
            st.error("无法打开视频源，请检查文件路径。")
            return

        perf_state = {"ema_cpu": None, "ema_fps": None, "ema_lat": None, "frame_count": 0, "alpha": 0.2}
        placeholders = st.session_state.get("_perf_placeholders", {})

        for frame_bgr, stats in gen:
            _update_perf_placeholders(stats, placeholders, perf_state)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            last_frame_rgb = frame_rgb
            video_container.image(frame_rgb, channels="RGB", use_container_width=True)

            if not calib_warning_shown and stats.get("calibration_fallback"):
                calib_placeholder.warning(
                    "⚠️ 未检测到踝关节，已回退至躯干校准。建议拍摄全身以获得更准的速度数据。"
                )
                calib_warning_shown = True

            rc = stats.get("reps", 0)
            if rc > last_rep_count and rc > 0:
                rep_velocities.append(stats.get("current_vel", 0.0))
                rom_pct = stats.get("rom_completion_pct")
                if rom_pct is not None:
                    rep_rom_completions.append(float(rom_pct))
                last_rep_count = rc
            final_stats = stats

            with metrics_container.container():
                st.caption(f"📊 Total Reps: {stats.get('reps', 0)}")
                m1, m2, m3, m4, m5 = st.columns(5)
                cur_mean = rep_velocities[-1] if rep_velocities else stats.get("current_vel", 0.0)
                best_mean = max(rep_velocities) if rep_velocities else stats.get("best_vel", 0.0)
                v_loss = ((best_mean - cur_mean) / best_mean * 100) if (rep_velocities and best_mean > 0) else stats.get("velocity_loss_pct", 0.0)
                with m1:
                    st.metric("Reps", stats.get("reps", 0))
                with m2:
                    st.metric("末次均速", f"{cur_mean:.3f} m/s")
                with m3:
                    st.metric("最佳均速", f"{best_mean:.3f} m/s")
                with m4:
                    st.metric("速度损失", f"{v_loss:.1f}%")
                with m5:
                    realtime_rom = stats.get("rom_completion_pct")
                    rom_val = float(realtime_rom) if realtime_rom is not None else 0.0
                    rom_str = f"{rom_val:.1f}%"
                    if rom_val >= 90:
                        st.markdown(
                            f'<span class="rom-ok">实时完成度: {rom_str} ✓</span>',
                            unsafe_allow_html=True,
                        )
                    elif rom_val < 85 and rom_val > 0:
                        st.markdown(
                            f'<span class="rom-low">实时完成度: {rom_str} ⚠ 幅度不足</span>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.metric("实时完成度", rom_str)
                avg_rom = sum(rep_rom_completions) / len(rep_rom_completions) if rep_rom_completions else 0.0
                avg_str = f"{avg_rom:.1f}%"
                if rep_rom_completions:
                    if avg_rom < 90:
                        st.markdown(
                            f'<span class="rom-low">平均完成度: {avg_str} ⚠ 幅度不足</span>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<span class="rom-ok">平均完成度: {avg_str}</span>',
                            unsafe_allow_html=True,
                        )

            up_pd = stats.get("pose_diag")
            if up_pd and pose_diag_on_b:
                with upload_pose_slot.container():
                    up1, up2, up3, up4 = st.columns(4)
                    up1.metric("技术评分", f"{up_pd.get('score', 100):.0f}")
                    up2.metric("膝踝比", f"{up_pd.get('knee_ratio', 1.0):.2f}")
                    up3.metric("躯干倾角", f"{up_pd.get('torso_angle', 0):.1f}°")
                    up4.metric("重心偏移", f"{up_pd.get('hip_x_drift_ratio', 0):.2%}")

            up_fat = None
            up_best = max(rep_velocities) if rep_velocities else stats.get("best_vel", 0)
            if len(rep_velocities) >= 3 and up_best > 0:
                up_fat = predict_fatigue(rep_velocities, up_best)
            up_goal = st.session_state.get("training_goal_select", "Strength")
            up_assess = assess_set_fatigue(rep_velocities, up_goal) if rep_velocities else None
            if up_assess or up_fat:
                with upload_fatigue_slot.container():
                    if up_assess:
                        up_colors = {"STOP": "#ef4444", "Fatigued": "#f97316", "Warning": "#eab308", "Optimal": "#22c55e"}
                        up_c = up_colors.get(up_assess.status, "#6b7280")
                        st.markdown(
                            f'<div style="padding:0.5rem 1rem; border-radius:8px; background:{up_c}22; border-left:4px solid {up_c}; font-weight:600;">'
                            f'{up_assess.message} (流失 {up_assess.velocity_loss_pct:.1f}% / 阈值 {up_assess.threshold_pct:.0f}%)</div>',
                            unsafe_allow_html=True,
                        )
                    if up_fat:
                        uf1, uf2, uf3 = st.columns(3)
                        uf1.metric("预测下把", f"{up_fat.predicted_next_v:.3f} m/s")
                        uf2.metric("预测衰减", f"{up_fat.predicted_loss_pct:.1f}%")
                        uf3.metric("置信度", f"{up_fat.confidence:.0%}")
                        if up_fat.failure_warning and not (up_assess and up_assess.status == "STOP"):
                            st.error(f"⚠️ {up_fat.message}")

            if rep_velocities:
                df_chart = pd.DataFrame(
                    {"速度 (m/s)": rep_velocities},
                    index=range(1, len(rep_velocities) + 1),
                )
                if HAS_PLOTLY:
                    up_baseline = max(rep_velocities) if rep_velocities else 0
                    up_thresh = 0.20 if up_goal == "Strength" else (0.35 if up_goal == "Hypertrophy" else 0.15)
                    cutoff = up_baseline * (1 - up_thresh) if up_baseline > 0 else 0
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(1, len(rep_velocities) + 1)),
                            y=rep_velocities,
                            mode="lines+markers",
                            name="向心速度",
                            line=dict(color="#22c55e", width=2),
                            marker=dict(size=8),
                        )
                    )
                    if cutoff > 0:
                        fig.add_hline(y=cutoff, line_dash="dash", line_color="#ef4444",
                                      annotation_text=f"{up_thresh*100:.0f}% 流失线")
                    if up_fat and up_fat.predicted_velocities:
                        pred_x_up = list(range(1, len(up_fat.predicted_velocities) + 1))
                        fig.add_trace(go.Scatter(
                            x=pred_x_up, y=up_fat.predicted_velocities,
                            mode="lines", name="AI 预测",
                            line=dict(color="#f59e0b", width=2, dash="dash"),
                        ))
                    fig.update_layout(
                        title="速度衰减曲线（实时）",
                        xaxis_title="Rep",
                        yaxis_title="速度 (m/s)",
                        template="plotly_dark",
                        height=320,
                        margin=dict(t=40, b=40),
                        uirevision="vbt-realtime",
                    )
                    chart_container.plotly_chart(
                        fig,
                        use_container_width=True,
                        theme="streamlit",
                        key=f"vbt_chart_ref_{time.time_ns()}",
                    )
                else:
                    chart_container.line_chart(df_chart, use_container_width=True)

            time.sleep(0.01)

            if stats.get("video_ended"):
                break

    except Exception as e:
        st.error(f"分析异常: {e}")
        st.code(traceback.format_exc(), language="text")
        logger.exception("本地视频分析异常")
    finally:
        if temp_path_to_clean:
            try:
                os.unlink(temp_path_to_clean)
            except OSError:
                pass

    if last_frame_rgb is not None:
        video_container.image(last_frame_rgb, channels="RGB", use_container_width=True)

    if final_stats is not None or rep_velocities:
        with summary_container.container():
            st.markdown("---")
            st.subheader("疲劳评估 — 总结报告")
            up_summary_goal = "Hypertrophy" if st.session_state.get("ai_mode_toggle", False) else "Strength"
            up_summary_assess = assess_set_fatigue(rep_velocities, up_summary_goal) if rep_velocities else None
            v_base = max(rep_velocities) if rep_velocities else (final_stats.get("best_vel", 0.0) if final_stats else 0.0)
            if up_summary_assess:
                if up_summary_assess.status == "STOP":
                    st.error(f"🚨 {up_summary_assess.message}")
                elif up_summary_assess.status == "Fatigued":
                    st.warning(f"⚠️ {up_summary_assess.message} (流失 {up_summary_assess.velocity_loss_pct:.1f}%)")
                elif up_summary_assess.status == "Warning":
                    st.warning(f"⚡ {up_summary_assess.message}")
                else:
                    st.success(f"✅ {up_summary_assess.message}")
                st.markdown(
                    f"- 基准速度: {up_summary_assess.baseline_vel:.3f} m/s | 末次速度: {rep_velocities[-1]:.3f} m/s | "
                    f"流失: {up_summary_assess.velocity_loss_pct:.1f}% (阈值 {up_summary_assess.threshold_pct:.0f}%)"
                )
            st.metric("总 Reps", len(rep_velocities))
            st.metric("最佳速度", f"{v_base:.3f} m/s")

            next_set = cur_set + 1
            st.info(f"🎉 第 {cur_set} 组分析完成！准备好第 {next_set} 组了吗？")
            if st.button(f"▶ 开始第 {next_set} 组", type="primary", key="next_set_btn"):
                st.session_state["current_set_number"] = next_set
                st.rerun()


def _render_session_review_tab(filter_user: str = "全部用户") -> None:
    """训练课复盘：按 session_id 聚合，展示组间速度趋势图。横轴为第几组 (Set Number)。"""
    st.markdown('<div class="nowrap"><b>训练课复盘</b> — 组间速度趋势</div>', unsafe_allow_html=True)

    # SQL：包含 id, ts, user_name, set_number, load_kg, mean_velocity, rom_pct
    base_sql = (
        "SELECT id, ts, session_id, set_number, "
        "v_mean AS mean_velocity, load_kg, rom_completion_pct AS rom_pct, pose_issues, "
        "COALESCE(user_name, 'qiqi') AS user_name "
        "FROM reps WHERE set_number IS NOT NULL ORDER BY ts"
    )
    if filter_user == "全部用户":
        df_all = _query_df(base_sql)
    else:
        uname = (filter_user or "qiqi").strip() or "qiqi"
        uname = "_".join(uname.split())
        df_all = _query_df(
            base_sql.replace("WHERE set_number", "WHERE user_name = ? AND set_number"),
            (uname,),
        )
    if df_all.empty:
        st.info("该用户暂无训练课复盘数据。" if filter_user != "全部用户" else "暂无训练课复盘数据。请在侧边栏配置「课号」和「组序号」后开始分析。")
        return

    sessions = sorted(df_all["session_id"].dropna().unique().tolist(), reverse=True)
    if not sessions:
        st.info("该用户暂无训练课复盘数据。")
        return

    selected_session = st.selectbox("选择训练课", sessions, index=0, key="review_session_select")
    df_session = df_all[df_all["session_id"] == selected_session].copy()
    if df_session.empty:
        st.info("该训练课暂无数据。")
        return

    # 过滤无效行，确保 set_number 和 mean_velocity 可用
    df_agg = df_session.dropna(subset=["set_number", "mean_velocity"])
    if df_agg.empty:
        st.info("该训练课无有效速度数据。")
        return

    # Named Aggregation：关键字 = 输出列名，元组 = (源列名, 聚合函数)
    try:
        set_agg = df_agg.groupby("set_number", dropna=False).agg(
            mean_velocity=("mean_velocity", "mean"),
            max_velocity=("mean_velocity", "max"),
            rep_count=("id", "count"),
            avg_rom_pct=("rom_pct", "mean"),
            load_kg=("load_kg", "first"),
        ).reset_index()
    except KeyError as e:
        st.error(f"数据库列名不匹配，请检查原始数据表结构。缺失列：{e}")
        return
    if "user_name" in df_agg.columns:
        user_map = df_agg.groupby("set_number", dropna=False)["user_name"].first()
        set_agg = set_agg.join(user_map, on="set_number")
    set_agg = set_agg.dropna(subset=["mean_velocity"]).sort_values("set_number", ascending=True).reset_index(drop=True)

    if set_agg.empty:
        st.info("聚合后无有效数据。")
        return

    st.markdown(f"##### 📊 课号: {selected_session}")

    c1, c2, c3 = st.columns(3)
    total_reps = int(set_agg["rep_count"].sum())
    total_sets = len(set_agg)
    avg_v_all = float(set_agg["mean_velocity"].mean()) if not set_agg.empty else 0.0
    c1.metric("总组数", total_sets)
    c2.metric("总动作数", total_reps)
    c3.metric("全课平均速度", f"{avg_v_all:.3f} m/s")

    st.markdown("**组间数据汇总**")
    rename_map = {
        "set_number": "组号",
        "mean_velocity": "平均速度(m/s)",
        "max_velocity": "最高速度(m/s)",
        "rep_count": "动作数",
        "avg_rom_pct": "平均ROM(%)",
        "load_kg": "负重(kg)",
    }
    if "user_name" in set_agg.columns:
        rename_map["user_name"] = "用户"
    display_df = set_agg.rename(columns=rename_map)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 横轴：第几组 (Set Number)，按 1, 2, 3... 升序
    if HAS_PLOTLY and len(set_agg) >= 2:
        x_vals = set_agg["set_number"].astype(int).tolist()
        y_vals = set_agg["mean_velocity"].tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name="平均向心速度",
            line=dict(color="#22c55e", width=3),
            marker=dict(size=10),
        ))
        if y_vals and y_vals[0] > 0:
            baseline_70 = y_vals[0] * 0.7
            fig.add_hline(y=baseline_70, line_dash="dash", line_color="#ef4444", annotation_text="70% 疲劳线")
        fig.update_layout(
            title="组间速度趋势图（疲劳积累监控）",
            xaxis_title="第几组 (Set Number)",
            yaxis_title="平均向心速度 (m/s)",
            template="plotly_dark",
            height=380,
            margin=dict(t=50, b=40),
            xaxis=dict(type="linear", dtick=1),
        )
        st.plotly_chart(fig, use_container_width=True, key="session_review_chart")

        v_first = y_vals[0]
        v_last = y_vals[-1]
        if v_first > 0:
            total_loss = (1 - v_last / v_first) * 100
            if total_loss > 30:
                st.error(f"⚠️ 全课速度衰减 {total_loss:.1f}%，疲劳积累严重，建议减少组数。")
            elif total_loss > 15:
                st.warning(f"⚡ 全课速度衰减 {total_loss:.1f}%，疲劳正在积累。")
            else:
                st.success(f"✅ 全课速度衰减 {total_loss:.1f}%，状态良好。")
    elif len(set_agg) < 2:
        st.caption("至少需要 2 组数据才能生成趋势图。")

    pose_col = df_session.get("pose_issues")
    if pose_col is not None:
        all_issues: list[str] = []
        for val in pose_col.dropna():
            all_issues.extend([t.strip() for t in str(val).split(",") if t.strip()])
        if all_issues:
            from collections import Counter
            issue_counts = Counter(all_issues)
            labels_cn = {"knee_valgus": "膝盖内扣", "torso_lean": "躯干代偿", "unstable": "重心不稳"}
            st.markdown("##### 🩺 技术短板分析")
            for issue, count in issue_counts.most_common():
                label = labels_cn.get(issue, issue)
                st.warning(f"**{label}** 出现 {count} 次")


def _render_video_management_tab() -> None:
    """训练视频管理：列表、预览、删除。"""
    from vbt_cv_engine import VIDEOS_DIR

    st.markdown('<div class="nowrap"><b>训练视频</b> — recordings/</div>', unsafe_allow_html=True)

    if not os.path.isdir(VIDEOS_DIR):
        st.info("暂无录制视频。开始实时采集并勾选「录制原始视频」即可自动保存。")
        return

    video_files = sorted(
        [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith((".mp4", ".avi", ".mov"))],
        reverse=True,
    )
    if not video_files:
        st.info("recordings/ 目录为空，暂无视频文件。")
        return

    st.caption(f"共 {len(video_files)} 个视频文件")

    selected = st.selectbox("选择视频预览", video_files, index=0, key="vid_preview_select")
    if selected:
        vid_path = os.path.join(VIDEOS_DIR, selected)
        st.video(vid_path)

        parts = selected.replace(".mp4", "").replace(".avi", "").replace(".mov", "").split("_")
        st.caption(f"文件: `{selected}` | 大小: {os.path.getsize(vid_path) / 1024 / 1024:.1f} MB")

    st.markdown("---")
    st.markdown("**批量删除**")

    to_delete = st.multiselect(
        "选择要删除的视频", video_files,
        key="vid_delete_multi",
    )

    if "pending_vid_delete" not in st.session_state:
        st.session_state["pending_vid_delete"] = False

    if st.session_state["pending_vid_delete"] and to_delete:
        st.warning(f"确认删除以下 {len(to_delete)} 个视频？此操作不可恢复！")
        for f in to_delete:
            st.text(f"  - {f}")
        ca, cb = st.columns(2)
        with ca:
            if st.button("是，确认删除", type="primary", key="vid_confirm_del"):
                deleted = 0
                for f in to_delete:
                    try:
                        os.remove(os.path.join(VIDEOS_DIR, f))
                        deleted += 1
                    except OSError as e:
                        st.error(f"删除失败 {f}: {e}")
                st.session_state["pending_vid_delete"] = False
                st.success(f"已删除 {deleted} 个视频文件。")
                st.rerun()
        with cb:
            if st.button("取消", key="vid_cancel_del"):
                st.session_state["pending_vid_delete"] = False
                st.rerun()
    elif to_delete:
        if st.button(f"删除选中的 {len(to_delete)} 个视频", type="secondary", key="vid_del_btn"):
            st.session_state["pending_vid_delete"] = True
            st.rerun()


def _render_data_management_tab(db_path: str, filter_user: str = "全部用户") -> None:
    """数据管理模块：层级视图（日期→组）、负重修正、组级删除、导出、自动清理。"""
    st.markdown('<div class="nowrap"><b>数据管理</b> — squat_gym.db</div>', unsafe_allow_html=True)
    ensure_db_safe(db_path)

    # ── 读取数据 ──
    try:
        if filter_user == "全部用户":
            df_reps = _query_df(
                "SELECT id, ts, COALESCE(user_name, 'qiqi') AS user_name, set_number, rep_count, "
                "v_mean, rom, rom_completion_pct, depth_offset_cm, load_kg, velocity_loss "
                "FROM reps ORDER BY ts ASC, set_number ASC, id ASC",
                db_path=db_path,
            )
        else:
            uname = (filter_user or "qiqi").strip() or "qiqi"
            uname = "_".join(uname.split())
            df_reps = _query_df(
                "SELECT id, ts, COALESCE(user_name, 'qiqi') AS user_name, set_number, rep_count, "
                "v_mean, rom, rom_completion_pct, depth_offset_cm, load_kg, velocity_loss "
                "FROM reps WHERE user_name = ? ORDER BY ts ASC, set_number ASC, id ASC",
                (uname,),
                db_path=db_path,
            )
    except Exception as e:
        st.error(f"读取训练记录失败: {e}")
        return

    if df_reps.empty:
        st.info("该用户暂无训练数据。" if filter_user != "全部用户" else "暂无历史数据。")
    else:
        # ── 层级视图：日期 + 组号 ──
        st.subheader("训练数据（按组浏览）")
        df_reps["date"] = df_reps["ts"].str[:10]
        df_reps["set_number"] = pd.to_numeric(df_reps["set_number"], errors="coerce")
        df_reps["set_label"] = df_reps["set_number"].fillna(0).astype(int)

        grouped = df_reps.groupby(["user_name", "date", "set_label"], sort=False)

        for (uname_grp, date_grp, set_grp), grp_df in grouped:
            n_reps = len(grp_df)
            load_val = grp_df["load_kg"].dropna()
            load_display = f"{load_val.iloc[0]:.1f}" if not load_val.empty else "—"
            avg_v = grp_df["v_mean"].mean()
            avg_v_str = f"{avg_v:.3f}" if pd.notna(avg_v) else "—"
            set_display = int(set_grp) if set_grp > 0 else "未分组"
            expander_title = f"📁 {date_grp} | 第 {set_display} 组 | 负重: {load_display} kg | {n_reps} Reps | 平均速度: {avg_v_str} m/s"
            if filter_user == "全部用户":
                expander_title = f"👤 {uname_grp} | {expander_title}"

            uid = f"{uname_grp}_{date_grp}_{set_grp}"

            with st.expander(expander_title, expanded=False):
                # ── 组内 Reps 详情 ──
                detail_df = grp_df[["id", "rep_count", "v_mean", "rom_completion_pct", "load_kg", "velocity_loss"]].rename(
                    columns={
                        "id": "ID",
                        "rep_count": "Rep#",
                        "v_mean": "速度(m/s)",
                        "rom_completion_pct": "ROM(%)",
                        "load_kg": "负重(kg)",
                        "velocity_loss": "速度损失(%)",
                    }
                )
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

                col_load, col_del = st.columns([2, 1])

                # ── 负重修正 ──
                with col_load:
                    cur_load = float(load_val.iloc[0]) if not load_val.empty else 0.0
                    new_load = st.number_input(
                        "修改本组负重 (kg)",
                        min_value=0.0,
                        value=cur_load,
                        step=2.5,
                        key=f"load_input_{uid}",
                    )
                    if st.button("💾 保存负重", key=f"save_load_{uid}"):
                        if set_grp > 0:
                            affected = update_set_load_kg(db_path, str(uname_grp), str(date_grp), int(set_grp), new_load)
                            st.success(f"已更新 {affected} 条记录的负重为 {new_load:.1f} kg")
                            st.rerun()
                        else:
                            st.warning("未分组的记录无法按组批量修改。")

                # ── 组级删除 ──
                with col_del:
                    st.markdown("<br>", unsafe_allow_html=True)
                    confirm_key = f"confirm_del_{uid}"
                    if confirm_key not in st.session_state:
                        st.session_state[confirm_key] = False

                    if st.session_state[confirm_key]:
                        st.warning(f"确认删除 {date_grp} 第 {set_display} 组的 {n_reps} 条记录？")
                        ca, cb = st.columns(2)
                        with ca:
                            if st.button("是，删除", type="primary", key=f"exec_del_{uid}"):
                                if set_grp > 0:
                                    deleted = delete_set_reps(db_path, str(uname_grp), str(date_grp), int(set_grp))
                                    st.session_state[confirm_key] = False
                                    st.success(f"已删除 {deleted} 条记录。")
                                    st.rerun()
                                else:
                                    st.warning("未分组的记录无法按组删除。")
                        with cb:
                            if st.button("取消", key=f"cancel_del_{uid}"):
                                st.session_state[confirm_key] = False
                                st.rerun()
                    else:
                        if st.button("🗑️ 删除此组", key=f"del_btn_{uid}"):
                            st.session_state[confirm_key] = True
                            st.rerun()

    # ── 导出数据 ──
    st.markdown("---")
    st.subheader("导出数据")
    if st.button("导出为 vbt_training_data.csv", key="export_btn"):
        try:
            df_export = _query_df(
                "SELECT * FROM reps ORDER BY id ASC",
                db_path=db_path,
            )
            if df_export.empty:
                st.warning("无数据可导出。")
            else:
                out_path = Path(__file__).resolve().parent / "vbt_training_data.csv"
                df_export.to_csv(out_path, index=False, encoding="utf-8-sig")
                st.success(f"已导出 {len(df_export)} 条记录至: {out_path}")
        except Exception as e:
            st.error(f"导出失败: {e}")

    # ── 自动清理无效记录 ──
    st.markdown("---")
    st.subheader("自动清理无效记录")
    st.caption("将删除: 分析日志中 reps_count=0 的记录；Reps 表中 ROM%<20% 的误触发数据")
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM reps WHERE rom_completion_pct IS NOT NULL AND rom_completion_pct < 20"
        )
        invalid_reps = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM analysis_logs WHERE reps_count = 0")
        invalid_logs = cur.fetchone()[0]
        conn.close()
    except Exception as e:
        st.error(f"统计失败: {e}")
        return

    st.write(f"待清理: Reps 无效记录(ROM%<20%)约 {invalid_reps} 条, 分析日志(0次)约 {invalid_logs} 条")
    pending_cleanup = st.session_state.get("pending_cleanup", False)
    if pending_cleanup:
        st.warning("⚠️ 确定要清理所有无效记录吗？此操作不可恢复。")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("是，执行清理", type="primary", key="execute_cleanup_btn"):
                try:
                    conn = sqlite3.connect(db_path)
                    cur = conn.cursor()
                    cur.execute(
                        "DELETE FROM reps WHERE rom_completion_pct IS NOT NULL AND rom_completion_pct < 20"
                    )
                    reps_deleted = cur.rowcount
                    cur.execute("DELETE FROM analysis_logs WHERE reps_count = 0")
                    logs_deleted = cur.rowcount
                    conn.commit()
                    conn.close()
                    st.session_state["pending_cleanup"] = False
                    st.success(f"已清理: Reps {reps_deleted} 条, 分析日志 {logs_deleted} 条。")
                    st.rerun()
                except Exception as e:
                    st.error(f"清理失败: {e}")
        with col_b:
            if st.button("取消", key="cancel_cleanup_btn"):
                st.session_state["pending_cleanup"] = False
                st.rerun()
    elif st.button("一键清理无效记录", type="primary", key="cleanup_btn"):
        st.session_state["pending_cleanup"] = True
        st.rerun()

    # ── 高级功能：历史数据批量校准 ──
    st.markdown("---")
    with st.expander("🛠️ 高级功能：历史数据批量校准 (Data Calibration)", expanded=False):
        st.info(
            "用于修复由于摄像头物理标定错误导致的历史速度异常。"
            "请谨慎操作，多次点击会导致数据被重复放大！"
        )
        if filter_user == "全部用户":
            st.warning("请先在上方筛选用户，选择要校准的目标用户后再操作。")
        else:
            multiplier = st.number_input(
                "速度乘数因子 (Velocity Multiplier)",
                min_value=0.1,
                max_value=1000.0,
                value=10.0,
                step=0.5,
                format="%.1f",
                key="velocity_multiplier",
            )
            confirm_calibration = st.checkbox(
                "我确认要修改该用户的所有历史速度数据",
                key="confirm_velocity_calibration",
            )
            if confirm_calibration:
                if st.button("⚠️ 立即批量修复", type="primary", key="exec_velocity_calibration"):
                    try:
                        uname = (filter_user or "qiqi").strip() or "qiqi"
                        uname = "_".join(uname.split())
                        conn = sqlite3.connect(db_path)
                        cur = conn.cursor()
                        cur.execute(
                            "UPDATE reps SET v_mean = v_mean * ? WHERE user_name = ? AND v_mean IS NOT NULL",
                            (float(multiplier), uname),
                        )
                        affected = cur.rowcount
                        conn.commit()
                        conn.close()
                        st.success(f"已将该用户 {affected} 条记录的 v_mean 乘以 {multiplier:.1f}。")
                        st.rerun()
                    except Exception as e:
                        st.error(f"批量校准失败: {e}")


def _render_legacy_tabs(user_name: str) -> None:
    """原有数据看板 Tab。"""
    distinct_users = get_distinct_user_names(DB_PATH)
    filter_options = ["全部用户"] + distinct_users
    default_idx = 0
    if user_name and user_name in distinct_users:
        default_idx = filter_options.index(user_name)
    filter_user = st.selectbox(
        "筛选用户",
        options=filter_options,
        index=default_idx,
        key="data_filter_user",
        help="选择要查看的训练数据所属用户",
    )
    if filter_user == "全部用户":
        df = _query_df(
            "SELECT ts, rep_count, v_mean, rom, rom_completion_pct, depth_offset_cm, load_kg, velocity_loss, COALESCE(set_number, 1) AS set_number, COALESCE(user_name, 'qiqi') AS user_name FROM reps ORDER BY ts DESC",
        )
    else:
        uname = (filter_user or "qiqi").strip() or "qiqi"
        uname = "_".join(uname.split())
        df = _query_df(
            "SELECT ts, rep_count, v_mean, rom, rom_completion_pct, depth_offset_cm, load_kg, velocity_loss, COALESCE(set_number, 1) AS set_number, COALESCE(user_name, 'qiqi') AS user_name FROM reps WHERE user_name = ? ORDER BY ts DESC",
            (uname,),
        )
    if df.empty and filter_user != "全部用户":
        st.info("该用户暂无训练数据。")
    tabs = st.tabs(["疲劳监测", "1RM 力量预测", "技术一致性", "数据管理", "训练课复盘", "训练视频"])
    with tabs[0]:
        st.markdown('<div class="nowrap"><b>疲劳监测</b>（基于侧边栏身高的动态物理标定）</div>', unsafe_allow_html=True)
        if df.empty:
            st.info("暂无训练数据。" if filter_user == "全部用户" else "该用户暂无训练数据。")
        else:
            fatigue_df = df.copy()
            fatigue_df["ts"] = pd.to_datetime(fatigue_df["ts"], errors="coerce")
            fatigue_df = fatigue_df.dropna(subset=["ts"]).sort_values("ts")
            fatigue_df["date_str"] = fatigue_df["ts"].dt.strftime("%Y-%m-%d")
            fatigue_df["date_display"] = fatigue_df["ts"].dt.strftime("%m-%d")
            fatigue_df["rep_in_set"] = fatigue_df.groupby(["date_str", "set_number"], dropna=False).cumcount() + 1
            load_val = fatigue_df["load_kg"].fillna(0).astype(float)
            fatigue_df["x_label"] = (
                fatigue_df["date_display"].astype(str) + " | " + load_val.astype(str) + "kg | "
                "第" + fatigue_df["set_number"].astype(int).astype(str) + "组 | "
                "第" + fatigue_df["rep_in_set"].astype(str) + "次"
            )
            if HAS_PLOTLY and not fatigue_df.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=fatigue_df["x_label"],
                        y=fatigue_df["velocity_loss"],
                        mode="lines+markers",
                        name="速度损失率 (%)",
                    )
                )
                fig.update_layout(
                    title="速度损失率趋势",
                    yaxis_title="速度损失率 (%)",
                    template="plotly_dark",
                    font=dict(family="Noto Sans CJK SC, Microsoft YaHei, PingFang SC"),
                )
                fig.update_xaxes(
                    type="category",
                    tickangle=-45,
                    title_text="训练节点 (日期 | 负重 | 组别 | 动作次序)",
                )
                st.plotly_chart(fig, use_container_width=True, key="history_fatigue_chart")
            else:
                st.dataframe(fatigue_df[["ts", "velocity_loss"]], use_container_width=True, hide_index=True)

    with tabs[1]:
        st.markdown('<div class="nowrap"><b>1RM 力量预测</b></div>', unsafe_allow_html=True)
        if df.empty:
            st.info("暂无可用于 1RM 回归的数据。")
        else:
            import numpy as np
            MIN_VELOCITY_1RM = 0.10  # 低于此速度视为无效数据（物理单位修复后正常范围 0.3~0.8 m/s）
            one_rm_base = df.dropna(subset=["load_kg", "v_mean"]).copy()
            one_rm_base = one_rm_base[(one_rm_base["load_kg"] > 0) & (one_rm_base["v_mean"] >= MIN_VELOCITY_1RM)]
            # 时间戳规范化：统一提取 YYYY-MM-DD
            one_rm_base["date_only"] = pd.to_datetime(one_rm_base["ts"], errors="coerce").dt.strftime("%Y-%m-%d")
            one_rm_base = one_rm_base.dropna(subset=["date_only"])
            # 负重类型清洗：统一为 float，避免 70 与 70.0 被判定为两种重量
            one_rm_base["load_kg"] = one_rm_base["load_kg"].astype(float)

            if one_rm_base.empty:
                st.info("负重与速度数据不足（速度 < 0.10 m/s 的异常数据已被过滤）。")
            else:
                # Task 1: 训练课选择器
                dates = sorted(one_rm_base["date_only"].dropna().unique(), reverse=True)
                session_options = ["汇总所有历史数据 (不推荐)"] + dates
                default_idx = 1 if dates else 0
                selected_session = st.selectbox(
                    "📅 选择用于预测的训练日期",
                    options=session_options,
                    index=min(default_idx, len(session_options) - 1),
                    key="1rm_session_select",
                    help="单日数据可得到更精准的 LVP 预测；汇总模式易受多日状态差异影响。",
                )

                # Task 2: 数据过滤
                if selected_session == "汇总所有历史数据 (不推荐)":
                    df_filtered = one_rm_base
                    data_source_label = "全部历史"
                    use_time_weighted = False
                else:
                    df_filtered = one_rm_base[one_rm_base["date_only"] == selected_session]
                    data_source_label = selected_session
                    use_time_weighted = True

                if df_filtered.empty:
                    st.warning(f"⚠ 所选日期「{selected_session}」无有效负重与速度数据。")
                else:
                    # 校验 A：原始选中日期的负重阶梯数量（未做速度过滤）
                    unique_loads_raw = df_filtered["load_kg"].astype(float).unique()
                    n_loads_raw = len(unique_loads_raw)

                    if n_loads_raw < 2:
                        st.warning("该训练日使用的负重阶梯不足（需至少 2 种不同重量）。")
                    else:
                        # 巅峰速度提取：每负荷取 max(v_mean)，仅最快 Rep 代表真实极限能力
                        grouped = df_filtered.groupby("load_kg", as_index=False)["v_mean"].max()
                        x = grouped["load_kg"].to_numpy(dtype=float)
                        y = grouped["v_mean"].to_numpy(dtype=float)
                        loads_today = x.tolist()
                        velocities_today = y.tolist()

                        # 构建历史数据（过去 30 天，排除选中日期）
                        history_rows: list[tuple[float, float, str]] = []
                        if use_time_weighted:
                            try:
                                target_dt = pd.to_datetime(selected_session)
                                cutoff = target_dt - pd.Timedelta(days=30)
                                df_hist = one_rm_base[
                                    (one_rm_base["date_only"] != selected_session)
                                    & (pd.to_datetime(one_rm_base["date_only"], errors="coerce") >= cutoff)
                                    & (pd.to_datetime(one_rm_base["date_only"], errors="coerce") < target_dt)
                                ]
                                hist_grouped = df_hist.groupby(["load_kg", "date_only"], as_index=False)["v_mean"].max()
                                for _, row in hist_grouped.iterrows():
                                    history_rows.append((float(row["load_kg"]), float(row["v_mean"]), str(row["date_only"])))
                            except Exception:
                                history_rows = []

                        one_rm: Optional[float] = None
                        slope: float = 0.0
                        intercept: float = 0.0
                        r_sq: float = 0.0
                        mvt_baseline = float(MVT_M_S)
                        historical_fusion_used = False

                        # 校验 B：负重足够但有效速度异常偏低
                        if np.max(y) < 0.01:
                            st.error(
                                "⚠ 检测到多组负重，但有效平均速度异常偏低（如 <0.01 m/s）。"
                                "请检查摄像头物理高度标定算法，当前速度数据不符合人类生物力学常识，无法进行准确的 LVP 线性回归。"
                            )
                        else:
                            if np.max(y) < 0.10:
                                st.warning(
                                    "⚠ 当前速度数据偏低（<0.1 m/s），预测仅供参考。建议校准摄像头物理高度标定。"
                                )

                            if use_time_weighted:
                                pred = predict_1rm_time_weighted(
                                    loads_today, velocities_today, selected_session,
                                    history_rows=history_rows if history_rows else None,
                                )
                                if pred is not None:
                                    slope = pred.slope
                                    intercept = pred.intercept
                                    r_sq = pred.r_squared
                                    mvt_baseline = pred.mvt_baseline
                                    one_rm = pred.predicted_1rm_kg
                                    historical_fusion_used = pred.historical_fusion_used
                                    if pred.uses_polynomial_fallback:
                                        st.caption("⚠ 线性拟合异常，已启用二次多项式保底曲线。")
                                else:
                                    slope, intercept = np.polyfit(x, y, 1)
                                    y_pred = slope * x + intercept
                                    ss_res = float(np.sum((y - y_pred) ** 2))
                                    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                                    r_sq = 1.0 - ss_res / ss_tot if abs(ss_tot) > 1e-12 else 0.0
                                    mvt_baseline = float(MVT_M_S)
                                    if slope < 0:
                                        one_rm = (mvt_baseline - intercept) / slope
                                        if one_rm <= 0 or one_rm > 1000:
                                            one_rm = None
                            else:
                                # 汇总模式：同样经 _clean_lvp_data 后拟合
                                pred = predict_1rm_time_weighted(
                                    loads_today, velocities_today,
                                    datetime.now().strftime("%Y-%m-%d"),
                                    history_rows=None,
                                )
                                if pred is not None:
                                    slope = pred.slope
                                    intercept = pred.intercept
                                    r_sq = pred.r_squared
                                    mvt_baseline = pred.mvt_baseline
                                    one_rm = pred.predicted_1rm_kg
                                    if pred.outliers_dropped:
                                        pass  # 下方统一展示 st.info
                                else:
                                    slope, intercept = np.polyfit(x, y, 1)
                                    y_pred = slope * x + intercept
                                    ss_res = float(np.sum((y - y_pred) ** 2))
                                    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                                    r_sq = 1.0 - ss_res / ss_tot if abs(ss_tot) > 1e-12 else 0.0
                                    mvt_baseline = float(MVT_M_S)
                                    if slope < 0:
                                        one_rm = (mvt_baseline - intercept) / slope
                                        if one_rm <= 0 or one_rm > 1000:
                                            one_rm = None

                            if historical_fusion_used:
                                st.success("🤖 单日数据方差过大，AI 已调取历史档案进行时间加权自学习，曲线已平滑修正。")
                            if pred is not None and pred.outliers_dropped:
                                st.info("✨ AI 已自动过滤未达最大发力意图的异常慢速点，LVP 曲线更加纯净。")

                            if slope >= 0 and one_rm is None:
                                st.warning("⚠ 数据异常：负重增加但速度未下降（斜率 ≥ 0），无法拟合有效的 1RM 模型。请增加不同负荷的训练组。")
                            elif one_rm is None or one_rm <= 0 or one_rm > 1000:
                                st.warning(f"⚠ 预测 1RM 异常（{one_rm or 'N/A'} kg），请检查数据质量或增加有效训练组。")
                            else:
                                st.metric("预测 1RM 极限重量", f"{one_rm:.1f} kg")
                                if r_sq < 0.65:
                                    st.markdown(
                                        f'<span style="color:#eab308; font-weight:600;">⚠ 数据线性度较差 (R² = {r_sq:.2f})，预测仅供参考，建议增加 70%-85% 负荷的有效组</span>',
                                        unsafe_allow_html=True,
                                    )

                            st.caption(f"模型: V = {slope:.5f} × Load + {intercept:.4f}　|　R² = {r_sq:.3f}　|　MVT = {mvt_baseline:.3f} m/s")

                            if HAS_PLOTLY:
                                chart_x = pred.loads if pred else x.tolist()
                                chart_y = pred.velocities if pred else y.tolist()
                                x_line = np.linspace(max(0, float(min(chart_x)) - 10), float(max(chart_x)) + 20, 80)
                                y_line = slope * x_line + intercept
                                lvp_legend = "LVP 曲线 (历史加权融合)" if historical_fusion_used else "LVP 回归线"
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=chart_x, y=chart_y, mode="markers",
                                    name="最佳发力点 (峰值提取)", marker=dict(size=10, color="#22c55e"),
                                ))
                                fig.add_trace(go.Scatter(
                                    x=x_line.tolist(), y=y_line.tolist(), mode="lines",
                                    name=lvp_legend, line=dict(color="#3b82f6", width=2),
                                ))
                                fig.add_hline(y=mvt_baseline, line_dash="dot", line_color="#ef4444",
                                              annotation_text=f"MVT = {mvt_baseline:.3f} m/s")
                                if one_rm is not None and 0 < one_rm <= 1000:
                                    fig.add_trace(go.Scatter(
                                        x=[one_rm], y=[mvt_baseline], mode="markers",
                                        name=f"预测 1RM ({one_rm:.1f} kg)",
                                        marker=dict(size=14, color="#ef4444", symbol="diamond"),
                                    ))
                                fig.update_layout(
                                    title=f"1RM 力量预测 (数据源: {data_source_label})　R² = {r_sq:.3f}",
                                    xaxis_title="负重 (kg)",
                                    yaxis_title="平均向心速度 (m/s)",
                                    template="plotly_dark",
                                    height=380,
                                    font=dict(family="Noto Sans CJK SC, Microsoft YaHei, PingFang SC"),
                                )
                                st.plotly_chart(fig, use_container_width=True, key="history_1rm_chart")

                            # Task 2: VBT 智能训练规划器（仅当 LVP 有效时展示）
                            if one_rm is not None and 0 < one_rm <= 1000 and slope < 0:
                                st.markdown("---")
                                with st.container():
                                    st.subheader("🎯 AI 智能训练处方 (基于今日 e1RM)")
                                    goal_options = [
                                        "最大力量 (Max Strength)",
                                        "肌肥大/增肌 (Hypertrophy)",
                                        "力量速度/爆发力 (Power)",
                                    ]
                                    goal = st.selectbox(
                                        "明日训练目标",
                                        options=goal_options,
                                        index=0,
                                        key="planner_goal",
                                    )
                                    if goal == "最大力量 (Max Strength)":
                                        pct_default, pct_min, pct_max = (90, 85, 95)
                                        loss_range = "15% - 20%"
                                    elif goal == "肌肥大/增肌 (Hypertrophy)":
                                        pct_default, pct_min, pct_max = (75, 70, 80)
                                        loss_range = "30% - 40%"
                                    else:
                                        pct_default, pct_min, pct_max = (75, 60, 85)
                                        loss_range = "10%"

                                    pct_1rm = st.slider(
                                        "目标负荷百分比 (%1RM)",
                                        min_value=pct_min,
                                        max_value=pct_max,
                                        value=pct_default,
                                        step=5,
                                        key="planner_pct",
                                    )
                                    target_load_raw = one_rm * (pct_1rm / 100)
                                    target_load = max(0.0, (target_load_raw // 2.5) * 2.5)
                                    target_vel = slope * target_load + intercept
                                    target_vel = round(target_vel, 2) if not np.isnan(target_vel) else 0.0

                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        st.metric("🏋️ 推荐杠铃负重", f"{target_load:.1f} kg")
                                    with c2:
                                        st.metric("⚡ 目标首发向心速度", f"{target_vel:.2f} m/s")
                                    with c3:
                                        st.metric("🛑 建议停组时机 (流失率阈值)", loss_range)

    with tabs[2]:
        st.markdown('<div class="nowrap"><b>技术一致性</b></div>', unsafe_allow_html=True)
        st.caption(f"筛选: {filter_user} | 样本数: {len(df)}")
        if not df.empty:
            if "rom_completion_pct" in df.columns:
                low_rom_mask = df["rom_completion_pct"].dropna() < 90
                if low_rom_mask.any():
                    st.markdown(
                        '<span class="rom-low">⚠ 幅度不足</span>（部分 Rep 完成度低于 90%）',
                        unsafe_allow_html=True,
                    )
            rename_tech = {
                "ts": "时间",
                "rep_count": "次数",
                "v_mean": "平均速度 (m/s)",
                "rom": "位移 ROM (m)",
                "rom_completion_pct": "完成度 (ROM %)",
                "depth_offset_cm": "下蹲深度偏移 (cm)",
                "load_kg": "负重 (kg)",
                "velocity_loss": "速度损失率 (%)",
            }
            if "user_name" in df.columns:
                rename_tech["user_name"] = "用户"
            tech_df = df.rename(columns=rename_tech)
            st.dataframe(tech_df, use_container_width=True, hide_index=True)

    with tabs[3]:
        _render_data_management_tab(DB_PATH, filter_user=filter_user)

    with tabs[4]:
        _render_session_review_tab(filter_user=filter_user)

    with tabs[5]:
        _render_video_management_tab()


def _render_ai_coach_section(user_name: str, load_kg: float) -> None:
    """AI 智能教练板块：LVP 可视化 + 实时卡片 + 建议。"""
    import numpy as np

    training_goal = st.session_state.get("training_goal_select", "Strength")
    training_mode = "hypertrophy" if training_goal == "Hypertrophy" else "strength"
    mode_label = {"Power": "爆发力", "Strength": "增力", "Hypertrophy": "肌肥大"}.get(training_goal, "增力")

    st.markdown("---")
    st.subheader(f"AI 智能教练 — {mode_label}模式")

    lvp = get_unified_lvp_prediction(DB_PATH, user_name)

    if lvp is None:
        st.info(f"AI 正在学习你的动作模式，请继续积累数据（需要至少 {MIN_DATA_POINTS} 个不同重量的训练记录）。")
        return

    rec_load = get_recommended_load(lvp, training_mode)

    uname = (user_name or "qiqi").strip() or "qiqi"
    uname = "_".join(uname.split())
    latest_v = _query_df(
        "SELECT v_mean FROM reps WHERE user_name = ? AND v_mean > 0.02 ORDER BY id DESC LIMIT 1",
        (uname,),
    )
    current_v = float(latest_v.iloc[0]["v_mean"]) if not latest_v.empty else 0.0

    latest_loss = _query_df(
        "SELECT velocity_loss FROM reps WHERE user_name = ? AND velocity_loss IS NOT NULL ORDER BY id DESC LIMIT 1",
        (uname,),
    )
    current_loss = float(latest_loss.iloc[0]["velocity_loss"]) if not latest_loss.empty else 0.0

    readiness = assess_daily_readiness(
        DB_PATH, user_name,
        current_load_kg=load_kg,
        current_first_set_velocity=current_v,
    )

    advice = get_training_advice(current_v, current_loss, training_mode, lvp, load_kg)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("预测 1RM", f"{lvp.predicted_1rm_kg:.1f} kg")
        if lvp.r_squared < 0.65:
            st.markdown(
                f'<span style="color:#eab308; font-size:0.8rem;">⚠ R²={lvp.r_squared:.2f}，仅供参考</span>',
                unsafe_allow_html=True,
            )
    with c2:
        status_emoji = {"optimal": "🟢", "normal": "🟡", "fatigued": "🔴"}.get(readiness.status, "⚪")
        st.metric("今日状态评分", f"{status_emoji} {readiness.score:.0f}/100")
    with c3:
        st.metric("推荐下一组重量", f"{rec_load:.1f} kg" if rec_load else "—")
    with c4:
        zone_colors = {"optimal": "green", "too_fast": "orange", "too_slow": "orange", "fatigued": "red"}
        zone_color = zone_colors.get(advice.velocity_zone, "gray")
        st.markdown(
            f'<div style="color:{zone_color}; font-weight:600; font-size:0.95rem;">{advice.message}</div>',
            unsafe_allow_html=True,
        )

    if HAS_PLOTLY and lvp.loads:
        with st.expander("LVP 负荷-速度曲线 (点击展开)", expanded=False):
            x_loads = np.asarray(lvp.loads, dtype=float)
            y_vels = np.asarray(lvp.velocities, dtype=float)

            x_min = max(0, float(x_loads.min()) - 10)
            x_max = max(float(x_loads.max()) + 20, lvp.predicted_1rm_kg + 10)
            x_line = np.linspace(x_min, x_max, 80)
            y_line = lvp.slope * x_line + lvp.intercept

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_loads.tolist(), y=y_vels.tolist(),
                mode="markers", name="最佳发力点 (峰值提取)",
                marker=dict(size=10, color="#22c55e"),
            ))
            fig.add_trace(go.Scatter(
                x=x_line.tolist(), y=y_line.tolist(),
                mode="lines", name="LVP 拟合线",
                line=dict(color="#3b82f6", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=[lvp.predicted_1rm_kg], y=[MVT_M_S],
                mode="markers", name=f"预测 1RM ({lvp.predicted_1rm_kg:.1f} kg)",
                marker=dict(size=14, color="#ef4444", symbol="diamond"),
            ))
            fig.add_hline(
                y=MVT_M_S, line_dash="dot", line_color="#ef4444",
                annotation_text=f"MVT = {MVT_M_S} m/s",
            )

            r2_color = "#22c55e" if lvp.r_squared >= 0.65 else "#eab308"
            fig.update_layout(
                title=f"个人负荷-速度曲线 (R² = {lvp.r_squared:.3f})",
                xaxis_title="负荷 (kg)",
                yaxis_title="平均向心速度 (m/s)",
                template="plotly_dark",
                height=380,
                font=dict(family="Noto Sans CJK SC, Microsoft YaHei, PingFang SC"),
            )
            st.plotly_chart(fig, use_container_width=True, key="lvp_chart")

            st.caption(
                f"模型: V = {lvp.slope:.5f} × Load + {lvp.intercept:.4f}　|　"
                f"数据点: {len(lvp.loads)}　|　R² = {lvp.r_squared:.3f}"
            )


def _render_system_footer() -> None:
    """页脚：树莓派 CPU/内存实时监控。"""
    if not HAS_PSUTIL:
        return
    try:
        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        mem_pct = mem.percent
        mem_used_mb = mem.used / (1024 * 1024)
        mem_total_mb = mem.total / (1024 * 1024)
        cpu_color = "#22c55e" if cpu_pct < 70 else ("#eab308" if cpu_pct < 90 else "#ef4444")
        mem_color = "#22c55e" if mem_pct < 70 else ("#eab308" if mem_pct < 90 else "#ef4444")
        st.markdown("---")
        st.markdown(
            f'<div style="text-align:center; font-size:0.8rem; color:#6b7280;">'
            f'🖥 CPU: <span style="color:{cpu_color}; font-weight:600;">{cpu_pct:.0f}%</span>　|　'
            f'💾 内存: <span style="color:{mem_color}; font-weight:600;">{mem_used_mb:.0f}/{mem_total_mb:.0f} MB ({mem_pct:.0f}%)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass


def render_streamlit_dashboard() -> None:
    """主入口：渲染完整看板。"""
    if st is None:
        raise RuntimeError("未安装 streamlit")

    ensure_db_safe(DB_PATH)

    st.set_page_config(page_title="VBT Command Center", layout="wide")
    st.title("VBT Command Center")
    _apply_dark_theme()

    mode, user_name, load_kg = _render_dual_mode_sidebar()

    _render_ai_coach_section(user_name, load_kg)

    if mode == "实时采集 (树莓派)":
        _render_mode_a_realtime()
    else:
        _render_mode_b_upload()

    st.markdown("---")
    st.subheader("数据看板")
    _render_legacy_tabs(user_name)

    _render_system_footer()


def main() -> None:
    if st is None:
        print("请安装 streamlit 后运行：streamlit run vbt_pro_coach_dashboard.py")
        return
    if "realtime_running" not in st.session_state:
        st.session_state["realtime_running"] = False
    render_streamlit_dashboard()


if __name__ == "__main__":
    main()
