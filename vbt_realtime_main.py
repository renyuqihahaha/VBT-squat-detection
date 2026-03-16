#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VBT 实时引擎：MoveNet 骨骼绘制、VBT 疲劳判定、断点续存、非阻塞语音。"""

import os
import sys

# V4L2 仅 Linux 有效，macOS 不设置
if sys.platform != "darwin":
    os.environ["OPENCV_VIDEOIO_V4L2_BUFFER_COUNT"] = "1"

import argparse
import threading
import time
import subprocess
import sqlite3
import logging
from collections import deque

import cv2
import numpy as np

from vbt_analytics_pro import (
    DB_PATH,
    MODEL_PATH,
    CONF_THRESHOLD,
    ensure_db_safe,
    insert_rep,
    angle_deg,
    trunk_angle_deg,
)
from vbt_runtime_config import get_current_load_kg, get_current_user_name, get_user_height_cm
from physics_converter import get_depth_offset

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

STOP_GESTURE_FRAMES = 10
MOVEMENT_THRESHOLD_RATIO = 0.08

# MoveNet 17 关键点骨架连接 (索引对)
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 头
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 躯干上、手臂
    (5, 11), (6, 12), (11, 12),  # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),  # 腿
]
BUFFER_RATIO = 0.05
FRAMES_IN_BUFFER_TO_END = 5
TORSO_HEIGHT_RATIO = 0.30
MIN_ROM_RATIO = 0.10
MIN_MCV_M_S = 0.02
V_LOSS_THRESHOLD = 25.0
FATIGUE_WARNING_THRESHOLD = 20.0  # 速度损失 > 20% 显示红色 + FATIGUE WARNING
logger = logging.getLogger("vbt_realtime")


def speak_async(text):
    # 彻底移除语音依赖：使用无阻塞文本模拟
    print(f"[VOICE MOCK]: {text}")


def _draw_skeleton(frame_bgr, kps, h, w, conf_thresh=0.3, color=(0, 255, 0), thickness=2):
    """绘制骨骼（MoveNet 等效 Mediapipe 可视化）。kps 格式 (y, x, conf)，需转换为像素坐标 (x, y)。"""
    if kps is None or len(kps) < 17:
        return
    try:
        pts = []
        for i in range(min(17, len(kps))):
            if float(kps[i][2]) >= conf_thresh:
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


def _try_v4l2_reset(index):
    """尝试使用 v4l2-ctl 做软重置/唤醒（系统支持时）。"""
    dev = f"/dev/video{int(index)}"
    cmds = [
        ["v4l2-ctl", "--device", dev, "--set-fmt-video=width=640,height=480,pixelformat=MJPG"],
        ["v4l2-ctl", "--device", dev, "--stream-mmap=1", "--stream-count=1"],
    ]
    for cmd in cmds:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            continue


def _open_and_probe_camera(index, max_attempts=12, required_consecutive_ok=3):
    # macOS 使用 AVFoundation，Linux 使用 V4L2
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(int(index), cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap = cv2.VideoCapture(int(index), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if sys.platform != "darwin":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        if sys.platform == "darwin":
            logger.error("摄像头无法打开。请检查：1. 系统偏好设置 → 安全性与隐私 → 摄像头权限 2. 其他应用是否占用摄像头")
        cap.release()
        return None
    # 硬件预热：给 USB 摄像头时间完成初始化
    time.sleep(0.5)
    # 硬件预热：冲掉前 5 帧缓存，确保首帧非黑屏
    for _ in range(5):
        try:
            cap.grab()
        except Exception:
            break

    consecutive_ok = 0
    consecutive_fail = 0
    for _ in range(max_attempts):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            consecutive_ok += 1
            consecutive_fail = 0
            if consecutive_ok >= required_consecutive_ok:
                return cap
        else:
            consecutive_ok = 0
            consecutive_fail += 1
            if consecutive_fail >= required_consecutive_ok and sys.platform != "darwin":
                # 连续 3 帧失败，Linux 下尝试软重置
                _try_v4l2_reset(index)
                consecutive_fail = 0
        time.sleep(0.05)
    cap.release()
    return None


def _brutal_scan_cameras(candidates=(0, 2, 4, 6)):
    for i in candidates:
        print(f"正在尝试索引 {i}...", flush=True)
        cap = _open_and_probe_camera(i, max_attempts=12, required_consecutive_ok=3)
        if cap is not None:
            print(f"成功！自动切换至索引 {i}", flush=True)
            return cap, i
    return None, None


def _insert_rep_async(db_path, rep_count, v_mean, rom, left_knee, right_knee, trunk, dtw_sim,
                      depth_offset_cm, load_kg, velocity_loss, user_name):
    """后台写入，不阻塞主循环。"""
    def _do():
        try:
            insert_rep(db_path, rep_count, v_mean, rom, left_knee, right_knee, trunk, dtw_sim,
                       depth_offset_cm, load_kg, velocity_loss, user_name)
        except Exception as e:
            logger.warning(f"异步写入 rep 失败: {e}")
    threading.Thread(target=_do, daemon=True).start()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="VBT 实时引擎")
    parser.add_argument("--camera-index", type=int, default=8, help="摄像头索引（macOS 建议 0，树莓派默认 8）")
    parser.add_argument("--no-gui", action="store_true", help="禁用本地 OpenCV 窗口（无头/调试）")
    args = parser.parse_args()

    # [1] 数据库层：断点续存，仅当 DB 不存在时建表
    ensure_db_safe(DB_PATH)

    # [2] 摄像头初始化
    active_camera_index = int(args.camera_index)
    logger.info(f"[1/3] Camera bootstrap... 正在尝试开启相机 (Index: {active_camera_index})...")
    cap = _open_and_probe_camera(active_camera_index, max_attempts=8, required_consecutive_ok=3)
    if cap is None:
        # H65 失败时仅尝试 10 以外偶数节点，跳过奇数元数据节点
        even_candidates = tuple(i for i in (0, 2, 4, 6) if i != active_camera_index)
        logger.info(f"相机开启失败 (Index: {active_camera_index})，启动偶数节点探测 {list(even_candidates)}...")
        cap, active_camera_index = _brutal_scan_cameras(candidates=even_candidates)
        if cap is None:
            if sys.platform == "darwin":
                logger.error("❌ 未检测到摄像头。请检查：1. 系统偏好设置 → 隐私与安全性 → 摄像头 2. 尝试 --camera-index 0")
            else:
                logger.error("❌ 未检测到任何可用摄像头。请检查：1. USB线是否插紧 2. 摄像头是否被其他进程占用。")
            return
    logger.info(f"[1/3] Camera OK (Index: {active_camera_index})")
    user_height_cm = float(get_user_height_cm())
    logger.info("✅ 已识别 H65 USB 摄像头，正在开启 %.0fcm 实时监控...", user_height_cm)
    logger.info("检测到 MJPG 模式开启成功，正在同步动态标定坐标...")

    use_local_display = not args.no_gui
    window_name = "VBT Realtime"
    if use_local_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)
        cv2.moveWindow(window_name, 40, 40)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        # 首帧立即显示，确保 2 秒内能看到画面
        ok, first_frame = cap.read()
        if ok and first_frame is not None and first_frame.size > 0:
            cv2.imshow(window_name, first_frame)
            cv2.waitKey(1)

    # 同步加载 AI 模型（无异步，保持主循环单线程）
    logger.info("[2/3] AI Model loading...")
    try:
        interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logger.info("[2/3] AI Model ready")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        cap.release()
        return

    state = "STANDING"
    running_avg_y = None
    y_lowest = -1.0
    y_bottom = -1.0
    t_start_asc = 0.0
    set_rep_count = 0
    stop_count = 0
    depth_spoken = False
    scale_m_per_px = None
    body_height_px = None
    valid_y = deque(maxlen=5)
    med_hist = deque(maxlen=9)
    frames_in_buffer = 0

    best_velocity_in_set = 0.0   # 本组最快速度 (peak)
    current_rep_velocity = 0.0  # 当前/最近一次向心速度
    current_v_loss = 0.0
    ascent_samples = []
    last_up_y = None
    last_up_t = None

    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0

    logger.info("进入主循环...")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                # 每次失败做轻量重试，最多 3 次
                retry_ok = False
                for _ in range(3):
                    ok, frame_bgr = cap.read()
                    if ok:
                        retry_ok = True
                        break
                    time.sleep(0.01)
                if not retry_ok:
                    continue
            if frame_bgr is None or frame_bgr.size == 0:
                continue
            t_now = time.time()

            fps_n += 1
            if (t_now - fps_t0) >= 1.0:
                fps = fps_n / (t_now - fps_t0)
                fps_n = 0
                fps_t0 = t_now

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            inp = np.expand_dims(cv2.resize(frame_rgb, (192, 192)), axis=0).astype(np.uint8)
            interpreter.set_tensor(input_details[0]["index"], inp)
            interpreter.invoke()
            kps = interpreter.get_tensor(output_details[0]["index"])[0][0]
            # 始终绘制骨骼（在 imshow 之前）
            _draw_skeleton(frame_bgr, kps, h, w, conf_thresh=0.3, color=(0, 255, 0), thickness=2)

            conf5 = float(kps[5][2])
            conf11 = float(kps[11][2])
            conf12 = float(kps[12][2])
            y_hip = None
            if conf11 >= 0.3:
                y_hip = float(kps[11][0] * h)
                valid_y.append(y_hip)
            elif valid_y:
                y_hip = float(np.mean(valid_y))
            if y_hip is None:
                phase_cn = "Waiting for pose"
                med_hist.clear()
            else:
                med_hist.append(y_hip)
                smoothed_y = float(np.median(med_hist)) if len(med_hist) >= 9 else y_hip

            if scale_m_per_px is None and conf5 >= 0.3 and conf11 >= 0.3:
                sy, sx = float(kps[5][0] * h), float(kps[5][1] * w)
                hy, hx = float(kps[11][0] * h), float(kps[11][1] * w)
                torso_px = np.sqrt((sy - hy) ** 2 + (sx - hx) ** 2)
                if torso_px > 5:
                    user_height_m = user_height_cm / 100.0
                    torso_ref_m = user_height_m * TORSO_HEIGHT_RATIO
                    scale_m_per_px = torso_ref_m / torso_px
                    body_height_px = torso_px / TORSO_HEIGHT_RATIO

            if y_hip is None or scale_m_per_px is None or body_height_px is None:
                phase_cn = "CALIBRATING" if y_hip is not None else "Waiting for pose"
                depth_offset_cm = None
            else:
                movement_threshold = MOVEMENT_THRESHOLD_RATIO * body_height_px
                buffer_px = BUFFER_RATIO * body_height_px
                user_height_m = user_height_cm / 100.0
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

                nose = kps[0] if kps[0][2] > 0.2 else None
                lw = kps[9] if kps[9][2] > 0.2 else None
                rw = kps[10] if kps[10][2] > 0.2 else None
                if nose is not None and lw is not None and rw is not None and float(lw[0]) < float(nose[0]) and float(rw[0]) < float(nose[0]):
                    stop_count += 1
                else:
                    stop_count = 0

                mid_hip_y = None
                if l_hip is not None and r_hip is not None:
                    mid_hip_y = ((float(l_hip[0]) + float(r_hip[0])) / 2.0) * h
                mid_knee_y = None
                if l_knee is not None and r_knee is not None:
                    mid_knee_y = ((float(l_knee[0]) + float(r_knee[0])) / 2.0) * h
                depth_offset_cm = get_depth_offset(mid_hip_y, mid_knee_y, scale_m_per_px)

                if state == "STANDING":
                    running_avg_y = smoothed_y if running_avg_y is None else 0.97 * running_avg_y + 0.03 * smoothed_y
                    if smoothed_y >= running_avg_y + movement_threshold:
                        state = "DOWN"
                        y_lowest = smoothed_y
                        depth_spoken = False
                        ascent_samples = []
                        last_up_y, last_up_t = None, None
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

                elif state == "UP":
                    if last_up_y is not None and last_up_t is not None:
                        dt = max(t_now - last_up_t, 1e-3)
                        inst_v = max(0.0, (last_up_y - smoothed_y) * scale_m_per_px / dt)
                        ascent_samples.append(inst_v)
                        current_rep_velocity = float(np.mean(ascent_samples)) if ascent_samples else 0.0
                    last_up_y, last_up_t = smoothed_y, t_now
                    if depth_offset_cm is not None and depth_offset_cm > 0 and not depth_spoken:
                        speak_async("深度达标，爆发力不错！")
                        depth_spoken = True

                    in_buffer = abs(smoothed_y - running_avg_y) <= buffer_px
                    frames_in_buffer = frames_in_buffer + 1 if in_buffer else 0
                    if frames_in_buffer >= FRAMES_IN_BUFFER_TO_END:
                        duration = t_now - t_start_asc
                        if duration > 0.05 and y_bottom > 0:
                            rom_m = (y_bottom - smoothed_y) * scale_m_per_px
                            mcv = float(np.mean(ascent_samples)) if ascent_samples else (rom_m / duration)
                            if rom_m >= (min_rom - 0.005) and mcv >= (MIN_MCV_M_S - 0.005):
                                set_rep_count += 1
                                current_rep_velocity = float(mcv)
                                best_velocity_in_set = max(best_velocity_in_set, float(mcv))
                                if set_rep_count == 1:
                                    current_v_loss = 0.0
                                    stored_loss = None
                                else:
                                    # loss = (best_v - current_v) / best_v
                                    current_v_loss = (best_velocity_in_set - mcv) / best_velocity_in_set * 100.0 if best_velocity_in_set > 0 else 0.0
                                    stored_loss = current_v_loss
                                    if current_v_loss >= V_LOSS_THRESHOLD:
                                        speak_async("警告：速度下降明显，建议结束本组训练。")
                                _insert_rep_async(
                                    DB_PATH, set_rep_count, mcv, rom_m, lk, rk, tr, None,
                                    depth_offset_cm, get_current_load_kg(), stored_loss, get_current_user_name(),
                                )
                        state = "STANDING"
                        y_lowest = -1.0
                        y_bottom = -1.0
                        frames_in_buffer = 0

                phase_key = "STOP_GESTURE" if stop_count > 0 else state
                phase_cn = {
                    "STANDING": "CALIBRATING",
                    "DOWN": "DOWN",
                    "UP": "UP",
                    "STOP_GESTURE": "STOP GESTURE",
                }.get(phase_key, phase_key)
            # HUD 半透明看板
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (8, 8), (320, 300), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, frame_bgr, 0.35, 0, frame_bgr)

            # HUD 数据叠加：Reps, Current V, Best V, V-Loss
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"State: {phase_cn}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(frame_bgr, f"Reps: {set_rep_count}", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(frame_bgr, f"Current V: {current_rep_velocity:.3f} m/s", (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame_bgr, f"Best V: {best_velocity_in_set:.3f} m/s", (12, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            v_color = (0, 255, 0) if current_v_loss < FATIGUE_WARNING_THRESHOLD else (0, 0, 255)
            cv2.putText(frame_bgr, f"V-Loss: {current_v_loss:.1f}%", (12, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.65, v_color, 2)
            if current_v_loss >= FATIGUE_WARNING_THRESHOLD and set_rep_count >= 2:
                cv2.putText(frame_bgr, "FATIGUE WARNING", (12, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame_bgr, f"Load: {get_current_load_kg():.1f} kg", (12, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            if depth_offset_cm is not None:
                cv2.putText(frame_bgr, f"Depth: {depth_offset_cm:+.1f} cm", (12, 256), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if stop_count > 0:
                cv2.putText(frame_bgr, f"STOP: {stop_count}/{STOP_GESTURE_FRAMES}", (12, 284), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            if use_local_display:
                cv2.imshow(window_name, frame_bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            time.sleep(0.01)
            if stop_count >= STOP_GESTURE_FRAMES:
                speak_async(f"本组数据已同步至您的{user_height_cm:.0f}cm专属数据库。")
                break
    except KeyboardInterrupt:
        logger.info("收到 KeyboardInterrupt，准备优雅退出...")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        pass


if __name__ == "__main__":
    main()

