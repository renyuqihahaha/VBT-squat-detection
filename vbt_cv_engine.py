#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBT Command Center 核心 CV 引擎：统一 process_squat_video(video_source)。
支持双数据源：摄像头索引 (int) 或视频文件路径 (str)。
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Generator, Optional, Union

import cv2
import numpy as np

from vbt_analytics_pro import (
    CONF_THRESHOLD,
    DB_PATH,
    MODEL_PATH,
    angle_deg,
    ensure_db_safe,
    insert_rep,
    trunk_angle_deg,
)
from vbt_runtime_config import get_current_load_kg, get_current_user_name, get_user_height_cm
from physics_converter import get_depth_offset, pixel_displacement_to_velocity_m_per_s
from vbt_fatigue_analyst import (
    compute_realtime_rom_percent,
    compute_standing_baseline,
    NOISE_THRESHOLD_PCT,
)
from vbt_ai_advisor import diagnose_pose, PoseDiagnosis
from vbt_perf_bridge import write_stats as _write_perf_stats

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# --- 常量 ---
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
MOVEMENT_THRESHOLD_RATIO = 0.08
BUFFER_RATIO = 0.05
FRAMES_IN_BUFFER_TO_END = 5
MIN_ROM_RATIO = 0.10
MIN_MCV_M_S = 0.02
DEPTH_MIN_M = 0.40   # 正常深蹲深度下限 (m)
DEPTH_MAX_M = 0.70   # 正常深蹲深度上限 (m)
FATIGUE_THRESHOLD = 0.7
TRAJECTORY_LEN = 80
ANATOMY_RATIO_SHOULDER_ANKLE = 0.80  # 肩中点到踝关节中心约占身高的 80%
ANATOMY_RATIO_HEAD_ANKLE = 0.90      # 鼻尖到踝关节中心约占身高的 90%
ANATOMY_RATIO_SHOULDER_HIP = 0.30    # 肩中点到髋关节中心约占身高的 30%
MIN_CALIB_SAMPLES = 10               # 标定所需最小样本数
CALIB_VISIBILITY_THRESHOLD = 0.5     # 关键点可见性阈值（正常）
CALIB_VISIBILITY_RELAXED = 0.3       # 标定阶段放宽可见度，应对杠铃片遮挡
CALIB_FORCE_FRAMES = 30              # 前 N 帧忽略 STANDING 检查，有人即采样
ANKLE_FALLBACK_FRAMES = 5            # 前 N 帧无踝则强制肩髋法
CALIB_TIMEOUT_FRAMES = 50            # 超时后强制使用默认比例尺
DEFAULT_FALLBACK_RATIO = 0.004       # 超时兜底：1 像素 = 0.004 米
FALLBACK_FRAME_LIMIT = 30            # 超过此帧数仍无踝关节则回退到躯干校准
MIN_DT_S = 0.001                     # dt 下限，防止零除
VELOCITY_NOISE_THRESHOLD = 0.05      # 速度 < 0.05 m/s 视为静止噪音
CALIB_FREEZE_FRAMES = 20             # 仅前 N 帧采集标定，之后永久锁定
MODEL_INPUT_SIZE = 192
PLATE_DIAMETER_M = 0.45              # 标准杠铃片直径 45cm
CAMERA_PROBE_ATTEMPTS = 12
CAMERA_CONSECUTIVE_OK = 3
logger = logging.getLogger("vbt_cv_engine")

# 边缘计算性能指标（供 Dashboard 实时读取）
_cv_engine_metrics: dict = {"latency_ms": None, "fps": None}

# 动态解剖标定提示（模块加载时打印一次）
print(
    "\n[VBT] 动态解剖自适应标定：肩踝 80%% 解剖常数，免疫相机距离/深度。"
    " 若历史数据异常，请前往 Dashboard → 数据管理 清理。\n"
)


def _try_v4l2_reset(index: int) -> None:
    """Linux V4L2 软重置（唤醒休眠 USB 摄像头）。"""
    dev = f"/dev/video{int(index)}"
    for cmd in [
        ["v4l2-ctl", "--device", dev, "--set-fmt-video=width=640,height=480,pixelformat=MJPG"],
        ["v4l2-ctl", "--device", dev, "--stream-mmap=1", "--stream-count=1"],
    ]:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
        except Exception:
            pass


def _open_camera_robust(index: int) -> Optional[cv2.VideoCapture]:
    """
    健壮的摄像头初始化：V4L2 后端 + MJPG + 640x480 + 单帧 buffer + 预热探测。
    移植自 vbt_realtime_main.py，确保树莓派 USB 摄像头稳定启动。
    """
    is_linux = sys.platform != "darwin"
    if is_linux:
        cap = cv2.VideoCapture(int(index), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(int(index), cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        logger.error("摄像头 %d 无法打开", index)
        cap.release()
        return None

    time.sleep(0.5)
    for _ in range(5):
        try:
            cap.grab()
        except Exception:
            break

    consecutive_ok = 0
    consecutive_fail = 0
    for _ in range(CAMERA_PROBE_ATTEMPTS):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            consecutive_ok += 1
            consecutive_fail = 0
            if consecutive_ok >= CAMERA_CONSECUTIVE_OK:
                logger.info("摄像头 %d 探测成功 (连续 %d 帧 OK)", index, consecutive_ok)
                return cap
        else:
            consecutive_ok = 0
            consecutive_fail += 1
            if consecutive_fail >= CAMERA_CONSECUTIVE_OK and is_linux:
                _try_v4l2_reset(index)
                consecutive_fail = 0
        time.sleep(0.05)

    logger.error("摄像头 %d 探测失败（%d 次尝试后无法获得 %d 连续帧）", index, CAMERA_PROBE_ATTEMPTS, CAMERA_CONSECUTIVE_OK)
    cap.release()
    return None


def _letterbox_preprocess(frame_rgb: np.ndarray, target: int = MODEL_INPUT_SIZE) -> tuple[np.ndarray, float, float, float, float]:
    """
    Letterbox 预处理：保持宽高比缩放 + 填充至 target×target。
    返回 (padded_image, offset_x, offset_y, scale, pad_ratio)。
    """
    h, w = frame_rgb.shape[:2]
    pad_ratio = max(w, h) / float(target)
    scale = target / max(w, h)
    w_new = int(w * scale)
    h_new = int(h * scale)
    resized = cv2.resize(frame_rgb, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    pad_w = target - w_new
    pad_h = target - h_new
    offset_x = pad_w / 2.0
    offset_y = pad_h / 2.0
    top, bottom = int(pad_h // 2), int(pad_h - pad_h // 2)
    left, right = int(pad_w // 2), int(pad_w - pad_w // 2)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, offset_x, offset_y, scale, pad_ratio


def _unpad_keypoint(y_norm: float, x_norm: float, h: int, w: int, offset_x: float, offset_y: float, scale: float) -> tuple[float, float]:
    """
    将 MoveNet 输出的归一化坐标 (0-1, 针对 192×192 canvas) 还原为原图像素坐标。
    y_norm, x_norm: 模型输出 [0,1]
    """
    pad_size = MODEL_INPUT_SIZE
    y_canvas = y_norm * pad_size
    x_canvas = x_norm * pad_size
    y_content = y_canvas - offset_y
    x_content = x_canvas - offset_x
    h_new = h * scale
    w_new = w * scale
    y_orig = y_content / h_new * h if h_new > 0 else 0.0
    x_orig = x_content / w_new * w if w_new > 0 else 0.0
    return y_orig, x_orig


def _kps_to_frame_coords(kps: np.ndarray, h: int, w: int, offset_x: float, offset_y: float, scale: float) -> np.ndarray:
    """将关键点从模型输出坐标转换为原图像素坐标。返回 shape (17, 3) 的数组，每行 [y_px, x_px, conf]。"""
    out = np.zeros_like(kps)
    for i in range(min(17, len(kps))):
        y_orig, x_orig = _unpad_keypoint(float(kps[i][0]), float(kps[i][1]), h, w, offset_x, offset_y, scale)
        out[i][0] = y_orig
        out[i][1] = x_orig
        out[i][2] = kps[i][2]
    return out


def _draw_skeleton(
    frame_bgr: np.ndarray,
    kps: np.ndarray,
    h: int,
    w: int,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    scale: float = 1.0,
    pixel_coords: bool = False,
    conf_thresh: float = 0.3,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    if kps is None or len(kps) < 17:
        return
    use_unpad = not pixel_coords and (offset_x != 0 or offset_y != 0 or scale != 1.0)
    try:
        pts = []
        for i in range(min(17, len(kps))):
            if float(kps[i][2]) >= conf_thresh:
                if pixel_coords:
                    x, y = int(kps[i][1]), int(kps[i][0])
                elif use_unpad:
                    y_orig, x_orig = _unpad_keypoint(float(kps[i][0]), float(kps[i][1]), h, w, offset_x, offset_y, scale)
                    x, y = int(x_orig), int(y_orig)
                else:
                    x = int(kps[i][1] * w)
                    y = int(kps[i][0] * h)
                if 0 <= x < w and 0 <= y < h:
                    pts.append((i, (x, y)))
                    cv2.circle(frame_bgr, (x, y), 4, color, -1)
        pts_dict = {idx: pt for idx, pt in pts}
        for (i1, i2) in SKELETON_EDGES:
            if i1 in pts_dict and i2 in pts_dict:
                cv2.line(frame_bgr, pts_dict[i1], pts_dict[i2], color, thickness)
    except (IndexError, TypeError, ValueError):
        pass


def _draw_trajectory(
    frame_bgr: np.ndarray,
    trajectory: list[tuple[float, float]],
    h: int,
    w: int,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> None:
    """绘制 Mid-hip 垂直位移轨迹线（身体重心轨迹）。"""
    if len(trajectory) < 2:
        return
    pts = []
    try:
        max_w, max_h = int(w), int(h)
    except Exception:
        return
    for item in trajectory:
        try:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                if isinstance(item[0], (int, float)):
                    x, y = float(item[0]), float(item[1])
                elif isinstance(item[0], (tuple, list)) and len(item[0]) >= 2:
                    x, y = float(item[0][0]), float(item[0][1])
                else:
                    continue
                if 0 <= x < max_w and 0 <= y < max_h:
                    pts.append((int(x), int(y)))
        except Exception:
            pass
    for i in range(1, len(pts)):
        cv2.line(frame_bgr, pts[i - 1], pts[i], color, thickness)


def _draw_bar_path(
    frame_bgr: np.ndarray,
    buffer: deque[tuple[int, int]],
    h: int,
    w: int,
) -> None:
    """绘制彗星效果轨迹线：杠铃/重心路径，越新越粗，末尾实心圆点。"""
    if len(buffer) < 2:
        return
    pts = list(buffer)
    # 使用亮黄色绘制轨迹连线，越新越粗（彗星尾巴效果）
    for i in range(1, len(pts)):
        thickness = int(np.interp(i, [0, len(pts)], [1, 4]))
        cv2.line(frame_bgr, pts[i - 1], pts[i], (0, 255, 255), thickness, cv2.LINE_AA)
    cv2.circle(frame_bgr, pts[-1], 6, (0, 255, 0), -1)


def _insert_rep_async(
    db_path: str,
    rep_count: int,
    v_mean: float,
    rom: float,
    left_knee: Optional[float],
    right_knee: Optional[float],
    trunk: Optional[float],
    dtw_sim: Optional[float],
    depth_offset_cm: Optional[float],
    load_kg: float,
    velocity_loss: Optional[float],
    user_name: str,
    rom_completion_pct: Optional[float] = None,
    set_number: Optional[int] = None,
    session_id: Optional[str] = None,
    user_height: Optional[float] = None,
    pose_issues: Optional[str] = None,
) -> None:
    def _do() -> None:
        try:
            insert_rep(
                db_path, rep_count, v_mean, rom, left_knee, right_knee, trunk, dtw_sim,
                depth_offset_cm, load_kg, velocity_loss, user_name, rom_completion_pct,
                set_number, session_id, user_height, pose_issues,
            )
        except Exception as e:
            logger.warning(f"异步写入 rep 失败: {e}")
    threading.Thread(target=_do, daemon=True).start()


def _render_hud(
    frame_bgr: np.ndarray,
    phase_cn: str,
    set_rep_count: int,
    inst_vel: float,
    best_mean_vel: float,
    current_v_loss: float,
    depth_offset_cm: Optional[float],
    fps: float,
    bar_shift_cm: Optional[float] = None,
    debug_ratio: Optional[float] = None,
    debug_dt: Optional[float] = None,
    debug_raw_dy_px: Optional[float] = None,
) -> None:
    """HUD 渲染：Inst Vel=瞬时速度，Best Mean=最佳均速(仅 Rep 完成后更新)。"""
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (8, 8), (340, 400), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame_bgr, 0.35, 0, frame_bgr)

    cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame_bgr, f"State: {phase_cn}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    cv2.putText(frame_bgr, f"Reps: {set_rep_count}", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    cv2.putText(frame_bgr, f"Inst Vel: {inst_vel:.3f} m/s", (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame_bgr, f"Best Mean: {best_mean_vel:.3f} m/s", (12, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    loss_rate = current_v_loss / 100.0
    v_color = (0, 255, 0) if loss_rate < FATIGUE_THRESHOLD else (0, 0, 255)
    cv2.putText(frame_bgr, f"Velocity Loss: {current_v_loss:.1f}%", (12, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.65, v_color, 2)
    if depth_offset_cm is not None:
        cv2.putText(frame_bgr, f"Depth: {depth_offset_cm:+.1f} cm", (12, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame_bgr, f"Load: {get_current_load_kg():.1f} kg", (12, 224), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    if bar_shift_cm is not None:
        shift_color = (0, 0, 255) if bar_shift_cm > 10.0 else (0, 255, 255)
        cv2.putText(frame_bgr, f"Path Shift: {bar_shift_cm:.1f} cm", (12, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.65, shift_color, 2)
    y_debug = 280
    if debug_ratio is not None:
        cv2.putText(frame_bgr, f"DEBUG_Ratio: {debug_ratio:.5f} m/px", (12, y_debug), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_debug += 20
    if debug_dt is not None:
        cv2.putText(frame_bgr, f"DEBUG_dt: {debug_dt:.3f} s", (12, y_debug), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_debug += 20
    if debug_raw_dy_px is not None:
        cv2.putText(frame_bgr, f"DEBUG_raw_dy_px: {debug_raw_dy_px:.1f} px", (12, y_debug), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def _render_fatigue_indicator(
    frame_bgr: np.ndarray,
    is_fatigue_70: bool,
    frame_n: int,
) -> None:
    """
    疲劳指示器：右上角显眼位置。
    正常：绿色；70% 疲劳：亮橙/红，闪烁 "Fatigue Warning: 70% reached"。
    """
    h, w = frame_bgr.shape[:2]
    x0, y0 = w - 320, 12
    if is_fatigue_70:
        flash = (frame_n // 15) % 2 == 0
        color = (0, 100, 255) if flash else (0, 50, 255)  # BGR: 橙/红
        cv2.rectangle(frame_bgr, (x0 - 4, y0 - 4), (w - 8, 52), (0, 0, 255), 2)
        cv2.putText(frame_bgr, "Fatigue Warning:", (x0, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if flash:
            cv2.putText(frame_bgr, "70% reached", (x0, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        cv2.rectangle(frame_bgr, (x0 - 4, y0 - 4), (w - 8, 42), (0, 255, 0), 2)
        cv2.putText(frame_bgr, "Fatigue: OK", (x0, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")


def _render_pose_warnings(
    frame_bgr: np.ndarray,
    kps: np.ndarray,
    diag: PoseDiagnosis,
    h: int,
    w: int,
    frame_n: int,
) -> None:
    """姿态异常时在画面上绘制红/黄加粗虚线和角落警告文字。"""
    if not diag.issues:
        return

    def _dashed_line(img: np.ndarray, p1: tuple, p2: tuple, color: tuple, thickness: int = 3, gap: int = 12) -> None:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = max(1, int(np.sqrt(dx * dx + dy * dy)))
        for i in range(0, dist, gap * 2):
            s = min(i, dist)
            e = min(i + gap, dist)
            sx = int(p1[0] + dx * s / dist)
            sy = int(p1[1] + dy * s / dist)
            ex = int(p1[0] + dx * e / dist)
            ey = int(p1[1] + dy * e / dist)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)

    flash = (frame_n // 10) % 2 == 0

    if diag.knee_valgus:
        lk_pt = (int(float(kps[13][1])), int(float(kps[13][0])))
        rk_pt = (int(float(kps[14][1])), int(float(kps[14][0])))
        la_pt = (int(float(kps[15][1])), int(float(kps[15][0])))
        ra_pt = (int(float(kps[16][1])), int(float(kps[16][0])))
        color = (0, 0, 255) if flash else (0, 80, 255)
        _dashed_line(frame_bgr, lk_pt, la_pt, color)
        _dashed_line(frame_bgr, rk_pt, ra_pt, color)

    if diag.torso_lean_warning:
        ls_pt = (int(float(kps[5][1])), int(float(kps[5][0])))
        lh_pt = (int(float(kps[11][1])), int(float(kps[11][0])))
        color = (0, 255, 255) if flash else (0, 200, 255)
        _dashed_line(frame_bgr, ls_pt, lh_pt, color)

    warn_y = h - 20
    for issue in reversed(diag.issues):
        labels = {
            "knee_valgus": "WARN: Knee Valgus",
            "torso_lean": "WARN: Torso Lean",
            "unstable": "WARN: Unstable",
        }
        txt = labels.get(issue, issue)
        if flash:
            cv2.putText(frame_bgr, txt, (10, warn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        warn_y -= 28


def _detect_plate_diameter_px(frame_bgr: np.ndarray) -> Optional[float]:
    """
    在画面中寻找最圆的物体（标准杠铃片），返回像素直径。
    使用 Hough 圆检测，优先返回尺寸合理（约 50-300px）的圆。
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape[:2]
    min_r = max(25, int(min(h, w) * 0.05))
    max_r = min(150, int(min(h, w) * 0.25))
    if min_r >= max_r:
        return None
    try:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=min_r * 2,
            param1=50, param2=30, minRadius=min_r, maxRadius=max_r,
        )
        if circles is None or len(circles[0]) == 0:
            return None
        circles = circles[0]
        best = max(circles, key=lambda c: c[2])
        return float(best[2] * 2)
    except Exception:
        return None


def _build_recording_filename(
    user_name: str, height_cm: float, load_kg: float, set_number: int,
) -> str:
    """生成录制文件名: {user_name}_{height}cm_{load}kg_set{n}_{timestamp}.mp4"""
    uname = (user_name or "qiqi").strip() or "qiqi"
    uname = "_".join(uname.split())
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{uname}_{height_cm:.0f}cm_{load_kg:.0f}kg_set{set_number}_{ts}.mp4"


def process_squat_video(
    video_source: Union[int, str],
    stop_flag: Optional[threading.Event] = None,
    set_number: Optional[int] = None,
    session_id: Optional[str] = None,
    user_height_cm: float = 175.0,
    record_video: bool = False,
    pose_diag_enabled: bool = True,
    use_plate_calibration: bool = False,
    plate_diameter_cm: Optional[float] = None,
    is_bodyweight: bool = False,
) -> Generator[tuple[np.ndarray, dict], None, None]:
    """
    统一处理入口：video_source 为 int（摄像头索引）或 str（视频文件路径）。
    record_video: 仅摄像头模式下生效，为 True 时录制原始帧到 recordings/ 目录。
    pose_diag_enabled: 姿态诊断开关，设为 False 可节省算力。
    use_plate_calibration: 使用标准杠铃片标定，替代解剖标定。
    plate_diameter_cm: 杠铃片直径 (cm)，由 UI 组装推导；None 时使用默认 45cm。
    is_bodyweight: 自重/徒手深蹲模式，锁定解剖标定并追踪髋部重心。
    stats 字典包含 'recording_path', 'pose_diag', 'rep_velocities' 键。
    """
    ensure_db_safe(DB_PATH)
    is_camera = isinstance(video_source, int)

    if is_camera:
        cap = _open_camera_robust(int(video_source))
        if cap is None:
            logger.error("摄像头 %s 初始化失败", video_source)
            return
    else:
        cap = cv2.VideoCapture(str(video_source))
        if not cap.isOpened():
            logger.error("无法打开视频文件: %s", video_source)
            return

    video_fps: Optional[float] = None
    if not is_camera:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps is None or video_fps <= 0:
            video_fps = 30.0
        logger.info("本地视频模式: FPS=%.1f", video_fps)

    try:
        interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        logger.error("模型加载失败: %s", e)
        cap.release()
        return

    # ── 动态比例尺：严禁硬编码，仅依赖 user_height_cm + standing_pixels ──
    user_height_cm = float(user_height_cm) if user_height_cm is not None else 0.0
    if user_height_cm < 100 or user_height_cm > 250:
        logger.warning("用户身高 %.1f cm 非法，使用默认 175 cm 计算比例尺", user_height_cm)
        user_height_cm = 175.0
    user_height_m = user_height_cm / 100.0

    # ── 动态解剖自适应标定状态 ──
    is_calibrated = False
    pixel_to_meter_ratio = 0.0
    calib_samples_primary: list[float] = []    # 肩-踝样本池
    calib_samples_secondary: list[float] = []  # 鼻-踝样本池
    calib_samples_tertiary: list[float] = []   # 肩-髋样本池
    plate_diameter_samples: list[float] = []   # 杠铃片直径样本池（兜底标定）
    force_tertiary_calib = False               # 前 5 帧无踝则强制肩髋法

    state = "STANDING"
    phase_cn = "Waiting for pose"
    running_avg_y: Optional[float] = None
    y_lowest = -1.0
    y_bottom = -1.0
    starting_height = -1.0
    t_start_asc = 0.0
    prev_inst_v: Optional[float] = None
    rom_completion_pct = 0.0
    last_completed_rom_pct: Optional[float] = None
    calibration_hip_samples: list[float] = []
    standing_height_locked: Optional[float] = None
    reference_max_displacement: Optional[float] = None
    set_rep_count = 0
    depth_spoken = False
    scale_m_per_px = None
    body_height_px = None
    calibration_fallback = False
    valid_y = deque(maxlen=5)
    med_hist = deque(maxlen=9)
    frames_in_buffer = 0
    best_velocity_in_set = 0.0
    current_rep_velocity = 0.0
    current_v_loss = 0.0
    ascent_samples = []
    last_up_y = None
    last_up_t = None
    last_frame_time: Optional[float] = None  # 真实时间戳，用于 dt 计算
    debug_ratio: Optional[float] = None
    debug_dt: Optional[float] = None
    debug_raw_dy_px: Optional[float] = None
    trajectory = deque(maxlen=TRAJECTORY_LEN)
    bar_path_buffer: deque[tuple[int, int]] = deque(maxlen=90)  # 杠铃/重心轨迹，约 3–5 秒
    last_bar_shift_cm: Optional[float] = None
    hip_x_history: list[float] = []
    rep_velocities_in_set: list[float] = []   # 每 Rep 的 rep_mean_vel (平均向心速度)
    last_rep_peak_vel: float = 0.0            # 上一 Rep 的瞬时峰值，供 stats 返回
    last_pose_diag = PoseDiagnosis()
    accumulated_pose_issues: list[str] = []
    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0
    frame_n = 0

    video_writer: Optional[cv2.VideoWriter] = None
    recording_path: Optional[str] = None
    if is_camera and record_video:
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        rec_name = _build_recording_filename(
            user_name=get_current_user_name(),
            height_cm=user_height_cm,
            load_kg=get_current_load_kg(),
            set_number=set_number or 1,
        )
        recording_path = os.path.join(VIDEOS_DIR, rec_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(recording_path, fourcc, 30.0, (640, 480))
        if video_writer.isOpened():
            logger.info("视频录制已启动: %s", recording_path)
        else:
            logger.warning("VideoWriter 初始化失败，跳过录制")
            video_writer = None
            recording_path = None

    try:
      while True:
        if stop_flag and stop_flag.is_set():
            break
        ok, frame_bgr = cap.read()
        if not ok:
            if is_camera:
                for _ in range(3):
                    ok, frame_bgr = cap.read()
                    if ok:
                        break
                    time.sleep(0.01)
            if not ok:
                if not is_camera:
                    best_mean = max(rep_velocities_in_set) if rep_velocities_in_set else 0.0
                    summary = {
                        "reps": set_rep_count,
                        "best_vel": best_mean,
                        "video_ended": True,
                    }
                    logger.info("本地视频结束，分析摘要: reps=%d, best_mean=%.3f m/s", set_rep_count, best_mean)
                    summary_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    summary_frame[:] = (30, 30, 30)
                    cv2.putText(summary_frame, "Video Analysis Complete", (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    cv2.putText(summary_frame, f"Total Reps: {set_rep_count}", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    best_mean_s = max(rep_velocities_in_set) if rep_velocities_in_set else 0.0
                    cv2.putText(summary_frame, f"Best Mean Vel: {best_mean_s:.3f} m/s", (80, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    best_mean_end = max(rep_velocities_in_set) if rep_velocities_in_set else 0.0
                    last_mean_end = rep_velocities_in_set[-1] if rep_velocities_in_set else 0.0
                    yield summary_frame, {
                        "video_ended": True,
                        "summary": summary,
                        "reps": set_rep_count,
                        "best_vel": best_mean_end,
                        "current_vel": last_mean_end,
                        "velocity_loss_pct": current_v_loss,
                        "phase": phase_cn,
                        "fps": fps,
                        "rom_completion_pct": last_completed_rom_pct if last_completed_rom_pct is not None else rom_completion_pct,
                    }
                break
        if frame_bgr is None or frame_bgr.size == 0:
            continue

        frame_start = time.time()
        t_now = frame_start
        if last_frame_time is not None:
            dt_frame = t_now - last_frame_time
            if dt_frame < MIN_DT_S:
                continue  # 跳过过近帧，防止零除
        last_frame_time = t_now
        frame_n += 1
        fps_n += 1
        if (t_now - fps_t0) >= 1.0:
            fps = fps_n / (t_now - fps_t0)
            fps_n = 0
            fps_t0 = t_now

        if video_writer is not None:
            video_writer.write(frame_bgr)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        padded, offset_x, offset_y, scale, _ = _letterbox_preprocess(frame_rgb, MODEL_INPUT_SIZE)
        inp = np.expand_dims(padded, axis=0).astype(np.uint8)
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        kps_raw = interpreter.get_tensor(output_details[0]["index"])[0][0]
        kps = _kps_to_frame_coords(kps_raw, h, w, offset_x, offset_y, scale)

        _draw_skeleton(frame_bgr, kps, h, w, pixel_coords=True, conf_thresh=0.3, color=(0, 255, 0), thickness=2)

        conf5 = float(kps[5][2])
        conf11 = float(kps[11][2])
        conf12 = float(kps[12][2])
        conf9 = float(kps[9][2]) if len(kps) > 9 else 0.0
        conf10 = float(kps[10][2]) if len(kps) > 10 else 0.0
        if is_bodyweight:
            tracked_y_raw = (float(kps[11][0]) + float(kps[12][0])) / 2.0 if (conf11 >= 0.3 and conf12 >= 0.3) else None
            tracked_x_raw = (float(kps[11][1]) + float(kps[12][1])) / 2.0 if (conf11 >= 0.3 and conf12 >= 0.3) else None
        else:
            if conf9 >= 0.3 and conf10 >= 0.3:
                tracked_y_raw = (float(kps[9][0]) + float(kps[10][0])) / 2.0
                tracked_x_raw = (float(kps[9][1]) + float(kps[10][1])) / 2.0
            elif conf11 >= 0.3:
                tracked_y_raw = float(kps[11][0])
                tracked_x_raw = float(kps[11][1])
            elif conf12 >= 0.3:
                tracked_y_raw = float(kps[12][0])
                tracked_x_raw = float(kps[12][1])
            elif valid_y:
                tracked_y_raw = float(np.mean(valid_y))
                tracked_x_raw = (float(kps[11][1]) + float(kps[12][1])) / 2.0 if (conf11 >= 0.3 and conf12 >= 0.3) else None
            else:
                tracked_y_raw = None
                tracked_x_raw = None
        if tracked_y_raw is not None:
            valid_y.append(tracked_y_raw)
        if tracked_y_raw is None:
            phase_cn = "Waiting for pose"
            med_hist.clear()
            smoothed_y = None
        else:
            med_hist.append(tracked_y_raw)
            smoothed_y = float(np.median(med_hist)) if len(med_hist) >= 9 else tracked_y_raw
        y_hip = tracked_y_raw

        if not is_calibrated:
            if is_bodyweight:
                conf5 = float(kps[5][2])
                conf6 = float(kps[6][2])
                la_conf = float(kps[15][2])
                ra_conf = float(kps[16][2])
                vis = CALIB_VISIBILITY_RELAXED
                shoulders_visible = conf5 >= vis or conf6 >= vis
                ankles_visible = la_conf >= vis or ra_conf >= vis
                if state == "STANDING" and shoulders_visible and ankles_visible and frame_n <= CALIB_FREEZE_FRAMES:
                    shoulder_ys = [float(kps[i][0]) for i in (5, 6) if float(kps[i][2]) >= vis]
                    ankle_ys = [float(kps[i][0]) for i in (15, 16) if float(kps[i][2]) >= vis]
                    if shoulder_ys and ankle_ys:
                        shoulder_y = float(np.mean(shoulder_ys))
                        ankle_y = float(np.mean(ankle_ys))
                        pixel_dist = abs(ankle_y - shoulder_y)
                        if pixel_dist > 30:
                            calib_samples_primary.append(pixel_dist)
                if len(calib_samples_primary) >= MIN_CALIB_SAMPLES:
                    median_pixels = float(np.median(calib_samples_primary))
                    pixel_to_meter_ratio = (user_height_m * ANATOMY_RATIO_SHOULDER_ANKLE) / median_pixels
                    scale_m_per_px = pixel_to_meter_ratio
                    body_height_px = median_pixels / ANATOMY_RATIO_SHOULDER_ANKLE
                    is_calibrated = True
                    logger.info(
                        "自重模式标定完成 (肩踝法): height=%.0fcm, median_px=%.1f, ratio=%.6f m/px",
                        user_height_cm, median_pixels, pixel_to_meter_ratio,
                    )
            elif not is_bodyweight and use_plate_calibration and frame_n <= CALIB_FREEZE_FRAMES:
                d_px = _detect_plate_diameter_px(frame_bgr)
                if d_px is not None and d_px > 30:
                    plate_diameter_samples.append(d_px)
                if len(plate_diameter_samples) >= MIN_CALIB_SAMPLES:
                    median_d = float(np.median(plate_diameter_samples))
                    plate_m = (plate_diameter_cm or 45.0) / 100.0
                    pixel_to_meter_ratio = plate_m / median_d
                    scale_m_per_px = pixel_to_meter_ratio
                    body_height_px = user_height_m / pixel_to_meter_ratio
                    is_calibrated = True
                    logger.info(
                        "杠铃片标定完成: D_px=%.1f, pixel_to_meter_ratio=%.6f m/px",
                        median_d, pixel_to_meter_ratio,
                    )

            if not is_bodyweight:
                conf5 = float(kps[5][2])
                conf6 = float(kps[6][2])
                conf11 = float(kps[11][2])
                conf12 = float(kps[12][2])
                nose_conf = float(kps[0][2])
                la_conf = float(kps[15][2])
                ra_conf = float(kps[16][2])
                vis = CALIB_VISIBILITY_RELAXED
                shoulders_visible = conf5 >= vis or conf6 >= vis
                ankles_visible = la_conf >= vis or ra_conf >= vis
                nose_visible = nose_conf >= vis
                hips_visible = conf11 >= vis or conf12 >= vis

                if frame_n <= ANKLE_FALLBACK_FRAMES and tracked_y_raw is not None and not ankles_visible:
                    force_tertiary_calib = True

                in_calib_window = frame_n <= max(CALIB_FREEZE_FRAMES, CALIB_FORCE_FRAMES)
                allow_sample = in_calib_window and (state == "STANDING" or frame_n <= CALIB_FORCE_FRAMES)
                person_detected = tracked_y_raw is not None

                if not is_calibrated and allow_sample and person_detected:
                    if not force_tertiary_calib and shoulders_visible and ankles_visible:
                        shoulder_ys = [float(kps[i][0]) for i in (5, 6) if float(kps[i][2]) >= vis]
                        ankle_ys = [float(kps[i][0]) for i in (15, 16) if float(kps[i][2]) >= vis]
                        if shoulder_ys and ankle_ys:
                            shoulder_y = float(np.mean(shoulder_ys))
                            ankle_y = float(np.mean(ankle_ys))
                            pixel_dist = abs(ankle_y - shoulder_y)
                            if pixel_dist > 30:
                                calib_samples_primary.append(pixel_dist)
                    elif not force_tertiary_calib and not shoulders_visible and nose_visible and ankles_visible:
                        ankle_ys = [float(kps[i][0]) for i in (15, 16) if float(kps[i][2]) >= vis]
                        if ankle_ys:
                            nose_y = float(kps[0][0])
                            ankle_y = float(np.mean(ankle_ys))
                            pixel_dist = abs(ankle_y - nose_y)
                            if pixel_dist > 20:
                                calib_samples_secondary.append(pixel_dist)
                    if force_tertiary_calib or (not ankles_visible and shoulders_visible and hips_visible):
                        shoulder_ys = [float(kps[i][0]) for i in (5, 6) if float(kps[i][2]) >= vis]
                        hip_ys = [float(kps[i][0]) for i in (11, 12) if float(kps[i][2]) >= vis]
                        if shoulder_ys and hip_ys:
                            shoulder_y = float(np.mean(shoulder_ys))
                            hip_y = float(np.mean(hip_ys))
                            pixel_dist = abs(hip_y - shoulder_y)
                            if pixel_dist > 5:
                                calib_samples_tertiary.append(pixel_dist)

            if frame_n > CALIB_TIMEOUT_FRAMES and not is_calibrated:
                pixel_to_meter_ratio = DEFAULT_FALLBACK_RATIO
                scale_m_per_px = pixel_to_meter_ratio
                body_height_px = user_height_m / pixel_to_meter_ratio
                is_calibrated = True
                logger.warning(
                    "标定超时 (frame %d)，强制使用默认比例尺 %.6f m/px",
                    frame_n, pixel_to_meter_ratio,
                )

            if not force_tertiary_calib and len(calib_samples_primary) >= MIN_CALIB_SAMPLES:
                median_pixels = float(np.median(calib_samples_primary))
                pixel_to_meter_ratio = (user_height_m * ANATOMY_RATIO_SHOULDER_ANKLE) / median_pixels
                scale_m_per_px = pixel_to_meter_ratio
                body_height_px = median_pixels / ANATOMY_RATIO_SHOULDER_ANKLE
                is_calibrated = True
                logger.info(
                    "动态解剖标定完成 (肩踝法): height=%.0fcm, median_px=%.1f, pixel_to_meter_ratio=%.6f m/px [168cm/720p 合理范围 0.003-0.005]",
                    user_height_cm, median_pixels, pixel_to_meter_ratio,
                )
            elif not force_tertiary_calib and len(calib_samples_secondary) >= MIN_CALIB_SAMPLES:
                median_pixels = float(np.median(calib_samples_secondary))
                pixel_to_meter_ratio = (user_height_m * ANATOMY_RATIO_HEAD_ANKLE) / median_pixels
                scale_m_per_px = pixel_to_meter_ratio
                body_height_px = median_pixels / ANATOMY_RATIO_HEAD_ANKLE
                is_calibrated = True
                logger.info(
                    "动态解剖标定完成 (头踝法): height=%.0fcm, median_px=%.1f, pixel_to_meter_ratio=%.6f m/px",
                    user_height_cm, median_pixels, pixel_to_meter_ratio,
                )
            elif len(calib_samples_tertiary) >= MIN_CALIB_SAMPLES * 2:
                median_pixels = float(np.median(calib_samples_tertiary))
                pixel_to_meter_ratio = (user_height_m * ANATOMY_RATIO_SHOULDER_HIP) / median_pixels
                scale_m_per_px = pixel_to_meter_ratio
                body_height_px = median_pixels / ANATOMY_RATIO_SHOULDER_HIP
                is_calibrated = True
                calibration_fallback = True
                logger.warning(
                    "动态解剖标定完成 (躯干兜底): height=%.0fcm, median_px=%.1f, pixel_to_meter_ratio=%.6f m/px",
                    user_height_cm, median_pixels, pixel_to_meter_ratio,
                )

        if not is_calibrated:
            if use_plate_calibration and len(plate_diameter_samples) > 0:
                calib_count, calib_target = len(plate_diameter_samples), MIN_CALIB_SAMPLES
                calib_method = "杠铃片"
            elif not force_tertiary_calib and len(calib_samples_primary) > 0:
                calib_count, calib_target = len(calib_samples_primary), MIN_CALIB_SAMPLES
                calib_method = "肩踝法"
            elif not force_tertiary_calib and len(calib_samples_secondary) > 0:
                calib_count, calib_target = len(calib_samples_secondary), MIN_CALIB_SAMPLES
                calib_method = "头踝法"
            elif len(calib_samples_tertiary) > 0 or force_tertiary_calib:
                calib_count, calib_target = len(calib_samples_tertiary), MIN_CALIB_SAMPLES * 2
                calib_method = "肩髋法"
            else:
                calib_count, calib_target = 0, MIN_CALIB_SAMPLES
                calib_method = "等待"

            ankles_vis = max(la_conf, ra_conf)
            shoulders_vis = max(conf5, conf6)
            debug_str = f"Debug: Ankles_Vis: {ankles_vis:.2f} | Shoulders_Vis: {shoulders_vis:.2f} | Method: {calib_method}"
            cv2.putText(frame_bgr, debug_str, (12, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

            if calib_count == 0:
                calib_text = "[ 🔍 正在寻找人体关键点... 请确保全身入镜 ]"
            else:
                calib_text = f"[ 🔄 标定中: {calib_count}/{calib_target} (正在使用 {calib_method}) ]"
            cv2.putText(frame_bgr, calib_text, (w // 2 - 200, h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            stats = {
                "reps": 0, "current_vel": 0.0, "best_vel": 0.0, "velocity_loss_pct": 0.0,
                "phase": "CALIBRATING", "fps": fps, "latency_ms": (time.time() - frame_start) * 1000,
                "rom_completion_pct": None, "pose_diag": None,
            }
            try:
                _write_perf_stats(fps if fps > 0 else 0.0, (time.time() - frame_start) * 1000)
            except Exception:
                pass
            yield frame_bgr, stats
            continue

        if y_hip is None or scale_m_per_px is None or body_height_px is None:
            phase_cn = "CALIBRATING" if y_hip is not None else "Waiting for pose"
            depth_offset_cm = None
        else:
            if state == "STANDING" and standing_height_locked is None and len(calibration_hip_samples) < MIN_CALIB_SAMPLES:
                calibration_hip_samples.append(smoothed_y)
                if len(calibration_hip_samples) >= MIN_CALIB_SAMPLES:
                    standing_height_locked = compute_standing_baseline(
                        calibration_hip_samples, num_frames=MIN_CALIB_SAMPLES
                    )
                    logger.info("站立高度已锁定: %.1f px", standing_height_locked)
            movement_threshold = MOVEMENT_THRESHOLD_RATIO * body_height_px
            buffer_px = BUFFER_RATIO * body_height_px
            min_rom = MIN_ROM_RATIO * user_height_m

            l_hip = kps[11] if conf11 > CONF_THRESHOLD else None
            r_hip = kps[12] if conf12 > CONF_THRESHOLD else None
            l_knee = kps[13] if kps[13][2] > CONF_THRESHOLD else None
            r_knee = kps[14] if kps[14][2] > CONF_THRESHOLD else None
            l_ank = kps[15] if kps[15][2] > CONF_THRESHOLD else None
            r_ank = kps[16] if kps[16][2] > CONF_THRESHOLD else None
            lk = angle_deg(l_hip, l_knee, l_ank)
            rk = angle_deg(r_hip, r_knee, r_ank)
            shoulder_mid = (kps[5][0], kps[5][1]) if conf5 > 0.2 else None
            hip_mid = None
            if l_hip is not None and r_hip is not None:
                hip_mid = ((float(l_hip[0]) + float(r_hip[0])) / 2.0, (float(l_hip[1]) + float(r_hip[1])) / 2.0)
            tr = trunk_angle_deg(shoulder_mid, hip_mid)

            mid_hip_x = (float(l_hip[1]) + float(r_hip[1])) / 2.0 if l_hip is not None and r_hip is not None else w // 2
            mid_hip_y = (float(l_hip[0]) + float(r_hip[0])) / 2.0 if l_hip is not None and r_hip is not None else h // 2
            trajectory.append((mid_hip_x, mid_hip_y))
            hip_x_history.append(mid_hip_x)
            if tracked_x_raw is not None and tracked_y_raw is not None:
                px, py = int(tracked_x_raw), int(tracked_y_raw)
                if 0 <= px < w and 0 <= py < h:
                    bar_path_buffer.append((px, py))

            if pose_diag_enabled and frame_n % 3 == 0:
                body_w_px = body_height_px * 0.35 if body_height_px else float(w) * 0.2
                last_pose_diag = diagnose_pose(kps, w, hip_x_history, body_w_px, state)
                if last_pose_diag.issues:
                    accumulated_pose_issues.extend(last_pose_diag.issues)

            mid_hip_y_val = (float(l_hip[0]) + float(r_hip[0])) / 2.0 if l_hip is not None and r_hip is not None else None
            mid_knee_y = (float(l_knee[0]) + float(r_knee[0])) / 2.0 if l_knee is not None and r_knee is not None else None
            depth_offset_cm = get_depth_offset(mid_hip_y_val, mid_knee_y, scale_m_per_px)

            if state == "STANDING":
                running_avg_y = smoothed_y if running_avg_y is None else 0.97 * running_avg_y + 0.03 * smoothed_y
                if smoothed_y >= running_avg_y + movement_threshold:
                    state = "DOWN"
                    y_lowest = smoothed_y
                    starting_height = standing_height_locked if standing_height_locked is not None else running_avg_y
                    depth_spoken = False
                    ascent_samples = []
                    bar_path_buffer.clear()
                    last_up_y, last_up_t = None, None
                    prev_inst_v = None
                    last_completed_rom_pct = None
                    if set_rep_count == 0:
                        best_velocity_in_set = 0.0
                        current_v_loss = 0.0

            elif state == "DOWN":
                if smoothed_y > y_lowest:
                    y_lowest = smoothed_y
                if smoothed_y < y_lowest - 2:
                    state = "UP"
                    y_bottom = y_lowest
                    t_start_asc = t_now
                    frames_in_buffer = 0
                    last_up_y, last_up_t = smoothed_y, t_now
                    prev_inst_v = None

            elif state == "UP":
                inst_v: Optional[float] = None
                if last_up_y is not None and last_up_t is not None and scale_m_per_px is not None:
                    dt = max(t_now - last_up_t, 1e-3)
                    raw_dy_px = abs(smoothed_y - last_up_y)
                    inst_v = pixel_displacement_to_velocity_m_per_s(raw_dy_px, scale_m_per_px, dt)
                    debug_ratio = scale_m_per_px
                    debug_dt = dt
                    debug_raw_dy_px = raw_dy_px
                    ascent_samples.append(inst_v)
                    current_rep_velocity = inst_v if inst_v >= 0.02 else 0.0
                    # 不再用瞬时峰值更新 best：best 仅来自 rep_mean_vel
                last_up_y, last_up_t = smoothed_y, t_now

                denom = starting_height - y_lowest
                if abs(denom) > 1e-6:
                    rom_completion = (smoothed_y - y_lowest) / denom
                    rom_completion_pct = min(100.0, max(0.0, rom_completion * 100.0))
                else:
                    rom_completion_pct = 0.0

                early_finish = False
                if inst_v is not None and prev_inst_v is not None and rom_completion_pct >= 85.0:
                    if inst_v < prev_inst_v:
                        early_finish = True
                if inst_v is not None:
                    prev_inst_v = inst_v

                if depth_offset_cm is not None and depth_offset_cm > 0 and not depth_spoken:
                    pass
                in_buffer = abs(smoothed_y - running_avg_y) <= buffer_px
                frames_in_buffer = frames_in_buffer + 1 if in_buffer else 0
                rep_done = early_finish or (frames_in_buffer >= FRAMES_IN_BUFFER_TO_END)
                if rep_done:
                    duration = t_now - t_start_asc
                    if duration > 0.05 and y_bottom > 0:
                        bottom_y_px = y_bottom
                        top_y_px = smoothed_y
                        depth_meters = abs(bottom_y_px - top_y_px) * scale_m_per_px
                        if depth_meters < DEPTH_MIN_M or depth_meters > DEPTH_MAX_M:
                            logger.warning(
                                "深度异常: %.3f m (合理范围 %.2f-%.2f m)，请检查 top_y/bottom_y 像素坐标",
                                depth_meters, DEPTH_MIN_M, DEPTH_MAX_M,
                            )
                        mcv = depth_meters / duration
                        rom_m = depth_meters
                        rep_peak_vel = float(max(ascent_samples)) if ascent_samples else 0.0
                        if rom_m >= (min_rom - 0.005) and mcv >= (MIN_MCV_M_S - 0.005):
                            set_rep_count += 1
                            current_rep_velocity = float(mcv)
                            rep_velocities_in_set.append(float(mcv))
                            best_velocity_in_set = max(rep_velocities_in_set)
                            last_rep_peak_vel = rep_peak_vel
                            if len(bar_path_buffer) >= 2 and scale_m_per_px is not None:
                                xs = [p[0] for p in bar_path_buffer]
                                bar_shift_px = max(xs) - min(xs)
                                last_bar_shift_cm = bar_shift_px * scale_m_per_px * 100.0
                            if set_rep_count == 1:
                                current_v_loss = 0.0
                                stored_loss = None
                            else:
                                loss_rate = (best_velocity_in_set - mcv) / best_velocity_in_set if best_velocity_in_set > 0 else 0.0
                                current_v_loss = loss_rate * 100.0
                                stored_loss = current_v_loss
                                if best_velocity_in_set > 0 and mcv < 0.7 * best_velocity_in_set:
                                    logger.warning("Fatigue 70%% reached: V_mean=%.3f < 0.7*Best_mean=%.3f", mcv, 0.7 * best_velocity_in_set)
                            rep_issue_tags = ",".join(sorted(set(accumulated_pose_issues))) if accumulated_pose_issues else None
                            accumulated_pose_issues.clear()
                            _insert_rep_async(
                                DB_PATH, set_rep_count, mcv, rom_m, lk, rk, tr, None,
                                depth_offset_cm, get_current_load_kg(), stored_loss, get_current_user_name(),
                                rom_completion_pct, set_number, session_id, user_height_cm,
                                rep_issue_tags,
                            )
                    if set_rep_count == 1:
                        reference_max_displacement = abs(starting_height - y_lowest)
                    last_completed_rom_pct = rom_completion_pct
                    state = "STANDING"
                    y_lowest = -1.0
                    y_bottom = -1.0
                    starting_height = -1.0
                    frames_in_buffer = 0
                    running_avg_y = smoothed_y
                    prev_inst_v = None
                    rom_completion_pct = 0.0
                    logger.info("Rep %d 完成，状态重置为 STANDING，准备检测下一动作", set_rep_count)

            phase_key = state
            phase_cn = {"STANDING": "CALIBRATING", "DOWN": "DOWN", "UP": "UP"}.get(phase_key, phase_key)

            if state == "STANDING":
                rom_completion_pct = 100.0
            elif (state == "DOWN" or state == "UP") and starting_height > 0 and y_lowest > 0:
                pct = compute_realtime_rom_percent(
                    starting_height, smoothed_y, y_lowest, noise_threshold_pct=NOISE_THRESHOLD_PCT
                )
                if pct is not None:
                    rom_completion_pct = pct

        _draw_bar_path(frame_bgr, bar_path_buffer, h, w)

        best_mean_vel = max(rep_velocities_in_set) if rep_velocities_in_set else 0.0
        inst_vel_for_hud = current_rep_velocity
        is_fatigue_70 = (
            best_mean_vel > 0 and len(rep_velocities_in_set) > 0
            and rep_velocities_in_set[-1] < 0.7 * best_mean_vel
        )
        _render_hud(frame_bgr, phase_cn, set_rep_count, inst_vel_for_hud, best_mean_vel, current_v_loss, depth_offset_cm, fps,
                    bar_shift_cm=last_bar_shift_cm,
                    debug_ratio=scale_m_per_px if scale_m_per_px is not None else debug_ratio,
                    debug_dt=debug_dt, debug_raw_dy_px=debug_raw_dy_px)
        _render_fatigue_indicator(frame_bgr, is_fatigue_70, frame_n)

        if pose_diag_enabled and last_pose_diag.issues:
            _render_pose_warnings(frame_bgr, kps, last_pose_diag, h, w, frame_n)

        stats_rom = last_completed_rom_pct if last_completed_rom_pct is not None else rom_completion_pct
        frame_end = time.time()
        dt = max(frame_end - frame_start, 1e-6)
        latency_ms_val = dt * 1000.0
        fps_val = fps if fps > 0 else (1.0 / dt)
        _cv_engine_metrics["latency_ms"] = latency_ms_val
        _cv_engine_metrics["fps"] = fps_val
        try:
            _write_perf_stats(fps_val, latency_ms_val)
        except Exception:
            pass

        last_rep_mean = rep_velocities_in_set[-1] if rep_velocities_in_set else None
        stats = {
            "reps": set_rep_count,
            "current_vel": float(last_rep_mean) if last_rep_mean is not None else current_rep_velocity,
            "best_vel": best_mean_vel,
            "velocity_loss_pct": current_v_loss,
            "phase": phase_cn,
            "fps": fps,
            "latency_ms": latency_ms_val,
            "is_fatigue_70": is_fatigue_70,
            "rom_completion_pct": stats_rom,
            "calibration_fallback": calibration_fallback,
            "recording_path": recording_path,
            "rep_mean_vel": last_rep_mean,
            "rep_peak_vel": last_rep_peak_vel if rep_velocities_in_set else None,
            "bar_shift_cm": last_bar_shift_cm,
            "pose_diag": {
                "knee_valgus": last_pose_diag.knee_valgus,
                "knee_ratio": last_pose_diag.knee_ratio,
                "torso_lean_warning": last_pose_diag.torso_lean_warning,
                "torso_angle": last_pose_diag.torso_angle,
                "stability_warning": last_pose_diag.stability_warning,
                "hip_x_drift_ratio": last_pose_diag.hip_x_drift_ratio,
                "issues": last_pose_diag.issues,
                "score": last_pose_diag.score,
            } if pose_diag_enabled else None,
            "rep_velocities": list(rep_velocities_in_set),
        }
        yield frame_bgr, stats

        if not is_camera and video_fps is not None and video_fps > 0:
            time.sleep(1.0 / video_fps)

    finally:
        _cv_engine_metrics["latency_ms"] = None
        _cv_engine_metrics["fps"] = None
        if video_writer is not None:
            video_writer.release()
            logger.info("视频录制已保存: %s", recording_path)
        cap.release()
        logger.info("摄像头/视频资源已释放")


def get_cv_engine_metrics() -> dict:
    """供 Dashboard 读取的视觉引擎性能指标。"""
    return dict(_cv_engine_metrics)
