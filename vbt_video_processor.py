#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频批处理脚本 vbt_video_processor.py — Left-Rear View, Single-Point (KP11 Left Hip)
- 计数仅用 Keypoint 11 (Left Hip) 垂直位移，其余关键点仅用于校准/角度/DTW
- 自适应基线 running_average_standing_y；Rep 开始：Y 下移 > 10% 身高；结束：5 帧连续进入缓冲带
- 抗抖：Moving Median(9)；conf<0.3 用最近 5 帧有效值平均插值
- 向心速度仅从 max_y 帧到回到缓冲带帧计算；调试输出 min_y/max_y/delta_y 及拒绝原因
- 目标：视频 ...331891 稳定 8 reps（不足则降低 movement_threshold）
"""

import argparse
import os
import sys
import cv2
import numpy as np
import time
import sqlite3
from collections import deque

from vbt_analytics_pro import (
    angle_deg,
    trunk_angle_deg,
    get_standard,
    dtw_similarity,
    DB_PATH,
    MODEL_PATH,
    CONF_THRESHOLD,
    init_db,
)

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

VIDEOS_DIR = "videos"
BATCH_TABLE = "batch_reps"

# --- Single-Point: KP11 Left Hip only for counting ---
TORSO_REF_M = 0.475
CONF_INTERPOLATE = 0.3                 # conf < 0.3 不置 0，用最近 5 帧有效 Y 平均插值
MEDIAN_WINDOW = 9                      # 左髋 Y 中值滤波 (window=9) 抗抖
VALID_Y_HISTORY = 5                    # 插值用最近有效帧数
MOVEMENT_THRESHOLD_RATIO = 0.08        # 相对身高：下移超过此比例才触发 rep 开始（0.10→0.08 提高检出）
BUFFER_RATIO = 0.05                    # 相对身高：缓冲带 = ±5% 身高 (px)
FRAMES_IN_BUFFER_TO_END = 5            # 连续 5 帧在缓冲带内则 rep 结束
# 按 ground truth 校准：331891/425005/867354 各 8，532794 为 6，637838 为 4，721883/807405 各 1
MIN_ROM_RATIO = 0.10                   # 有效 rep：ROM > 10% 身高（使 10.8cm 等边界通过）
MIN_MCV_M_S = 0.02                     # MCV 下限降至 0.02，避免 0.03–0.08 的慢速有效 rep 被拒


def init_batch_table(db_path=DB_PATH):
    """创建批处理结果表，含 barbell_path_y（杠铃 Y 序列）供 DTW。"""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS batch_reps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            rep_no INTEGER NOT NULL,
            v_mean REAL,
            min_knee_angle REAL,
            max_trunk_angle REAL,
            dtw_similarity REAL,
            barbell_path_y BLOB
        )
    """)
    try:
        cur.execute("ALTER TABLE batch_reps ADD COLUMN barbell_path_y BLOB")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def insert_batch_rep(db_path, filename, rep_no, v_mean, min_knee_angle, max_trunk_angle, dtw_similarity, barbell_path_y=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO batch_reps (filename, rep_no, v_mean, min_knee_angle, max_trunk_angle, dtw_similarity, barbell_path_y)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (filename, rep_no, v_mean, min_knee_angle, max_trunk_angle, dtw_similarity, barbell_path_y))
    conn.commit()
    conn.close()


def delete_batch_reps_by_filename(db_path, filename):
    """删除该 filename 在 batch_reps 中的全部记录，用于单视频重跑时先清空再补充。"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM batch_reps WHERE filename = ?", (filename,))
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    return deleted


def init_interpreter():
    """加载模型并返回 interpreter、input_details、output_details，供基准录入或批处理使用 [cite: 2026-01-31]。"""
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def process_video(video_path, interpreter, input_details, output_details, standard_seq=None, return_seq=False):
    """
    Single-Point: 仅 KP11 (Left Hip) 计数；自适应基线；Median(9)；conf<0.3 用最近 5 帧有效 Y 插值；
    Rep 开始：Y 下移 > movement_threshold（% 身高）；结束：连续 5 帧在缓冲带内；
    MCV 仅从 max_y 帧到回到缓冲带；调试输出 min_y/max_y/delta_y 及拒绝原因。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    h, w = None, None
    state = "STANDING"
    running_average_standing_y = None
    y_lowest = -1
    y_bottom = -1
    t_start_asc = 0
    rep_count = 0
    current_rep_sequence = []
    barbell_path_y_list = []
    first_rep_sequence = None
    min_left_knee = 180.0
    min_right_knee = 180.0
    max_trunk = -1.0
    results = []
    scale_m_per_px = None
    body_height_px = None
    lifter_height_m = None
    last_good_y_shld = None
    last_good_y_hip = None
    last_good_shld_x = None
    last_good_hip_x = None
    valid_y_deque = deque(maxlen=VALID_Y_HISTORY)
    median_hist = deque(maxlen=MEDIAN_WINDOW)
    frames_in_buffer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_curr = time.time()
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if h is None:
            h, w = frame.shape[0], frame.shape[1]

        input_data = np.expand_dims(cv2.resize(frame, (192, 192)), axis=0).astype(np.uint8)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        keypoints = interpreter.get_tensor(output_details[0]["index"])[0][0]

        conf5 = float(keypoints[5][2])
        conf11 = float(keypoints[11][2])
        if conf5 >= 0.3:
            last_good_y_shld = keypoints[5][0] * h
            last_good_shld_x = keypoints[5][1] * w
        if conf11 >= CONF_INTERPOLATE:
            raw_y_hip = keypoints[11][0] * h
            valid_y_deque.append(raw_y_hip)
            last_good_y_hip = raw_y_hip
            last_good_hip_x = keypoints[11][1] * w
            y_to_use = raw_y_hip
        else:
            y_to_use = float(np.mean(valid_y_deque)) if valid_y_deque else None
        if y_to_use is None:
            continue

        median_hist.append(y_to_use)
        smoothed_y = float(np.median(median_hist)) if len(median_hist) >= MEDIAN_WINDOW else y_to_use

        if scale_m_per_px is None and last_good_y_shld is not None and last_good_y_hip is not None and last_good_shld_x is not None and last_good_hip_x is not None:
            sy, sx = last_good_y_shld, last_good_shld_x
            hy, hx = last_good_y_hip, last_good_hip_x
            dist_px = np.sqrt((sy - hy) ** 2 + (sx - hx) ** 2)
            if dist_px > 5:
                scale_m_per_px = TORSO_REF_M / dist_px
                body_height_px = 2.0 * dist_px
                lifter_height_m = body_height_px * scale_m_per_px

        if scale_m_per_px is None or body_height_px is None:
            continue

        movement_threshold_px = MOVEMENT_THRESHOLD_RATIO * body_height_px
        buffer_px = BUFFER_RATIO * body_height_px

        l_hip = keypoints[11] if conf11 >= CONF_THRESHOLD else None
        l_knee = keypoints[13] if keypoints[13][2] > CONF_THRESHOLD else None
        r_knee = keypoints[14] if keypoints[14][2] > CONF_THRESHOLD else None
        l_ankle = keypoints[15] if keypoints[15][2] > CONF_THRESHOLD else None
        r_ankle = keypoints[16] if keypoints[16][2] > CONF_THRESHOLD else None
        r_hip = keypoints[12] if keypoints[12][2] > CONF_THRESHOLD else None
        left_knee_deg = angle_deg(l_hip, l_knee, l_ankle) if l_hip is not None else None
        right_knee_deg = angle_deg(r_hip, r_knee, r_ankle) if r_hip is not None else None
        shoulder_mid = (keypoints[5][0], keypoints[5][1]) if conf5 >= 0.2 else None
        hip_mid = (keypoints[11][0], keypoints[11][1]) if conf11 >= 0.2 else None
        trunk_deg = trunk_angle_deg(shoulder_mid, hip_mid)
        lk = left_knee_deg if left_knee_deg is not None else 0.0
        rk = right_knee_deg if right_knee_deg is not None else 0.0
        tr = trunk_deg if trunk_deg is not None else 0.0

        if state == "STANDING":
            if running_average_standing_y is None:
                running_average_standing_y = smoothed_y
            else:
                running_average_standing_y = 0.97 * running_average_standing_y + 0.03 * smoothed_y
            if smoothed_y >= running_average_standing_y + movement_threshold_px:
                state = "DOWN"
                y_lowest = smoothed_y
                barbell_path_y_list = [smoothed_y]
                current_rep_sequence = [(lk, rk, tr)]
                min_left_knee = 180.0
                min_right_knee = 180.0
                max_trunk = -1.0
                if left_knee_deg is not None:
                    min_left_knee = min(min_left_knee, left_knee_deg)
                if right_knee_deg is not None:
                    min_right_knee = min(min_right_knee, right_knee_deg)
                if trunk_deg is not None:
                    max_trunk = max(max_trunk, trunk_deg)

        elif state == "DOWN":
            if smoothed_y > y_lowest:
                y_lowest = smoothed_y
            barbell_path_y_list.append(smoothed_y)
            current_rep_sequence.append((lk, rk, tr))
            if left_knee_deg is not None:
                min_left_knee = min(min_left_knee, left_knee_deg)
            if right_knee_deg is not None:
                min_right_knee = min(min_right_knee, right_knee_deg)
            if trunk_deg is not None:
                max_trunk = max(max_trunk, trunk_deg)
            if smoothed_y < y_lowest - 2:
                state = "UP"
                t_start_asc = t_curr
                y_bottom = y_lowest
                frames_in_buffer = 0

        elif state == "UP":
            barbell_path_y_list.append(smoothed_y)
            current_rep_sequence.append((lk, rk, tr))
            if left_knee_deg is not None:
                min_left_knee = min(min_left_knee, left_knee_deg)
            if right_knee_deg is not None:
                min_right_knee = min(min_right_knee, right_knee_deg)
            if trunk_deg is not None:
                max_trunk = max(max_trunk, trunk_deg)
            in_buffer = abs(smoothed_y - running_average_standing_y) <= buffer_px
            if in_buffer:
                frames_in_buffer += 1
            else:
                frames_in_buffer = 0
            if frames_in_buffer >= FRAMES_IN_BUFFER_TO_END:
                duration = t_curr - t_start_asc
                min_y_rep = running_average_standing_y
                max_y_rep = y_bottom
                delta_y_px = max_y_rep - min_y_rep
                delta_y_m = delta_y_px * scale_m_per_px
                if duration > 0.05 and y_bottom > 0:
                    rom_m = (y_bottom - smoothed_y) * scale_m_per_px
                    mcv = rom_m / duration
                    min_rom = MIN_ROM_RATIO * lifter_height_m
                    min_knee_angle = min(min_left_knee, min_right_knee)
                    if min_knee_angle > 179:
                        min_knee_angle = None
                    if max_trunk < 0:
                        max_trunk = None
                    valid = rom_m >= (min_rom - 0.005) and mcv >= (MIN_MCV_M_S - 0.005)
                    print(f"  [DEBUG] min_y={min_y_rep:.1f} max_y={max_y_rep:.1f} delta_y_px={delta_y_px:.1f} delta_y_m={delta_y_m*100:.1f}cm ROM={rom_m*100:.1f}cm MCV={mcv:.2f} m/s")
                    if valid:
                        rep_count += 1
                        dtw_sim = None
                        if standard_seq and current_rep_sequence:
                            dtw_sim = dtw_similarity(current_rep_sequence, standard_seq)
                            dtw_sim = round(dtw_sim, 4) if dtw_sim is not None else None
                        path_blob = np.array(barbell_path_y_list, dtype=np.float32).tobytes()
                        results.append((rep_count, mcv, min_knee_angle, max_trunk, dtw_sim, path_blob))
                        if return_seq and rep_count == 1 and len(current_rep_sequence) > 5:
                            first_rep_sequence = list(current_rep_sequence)
                        print(f"  ✅ 有效 rep {rep_count} | MCV: {mcv:.2f} m/s | ROM: {rom_m*100:.0f}cm")
                    else:
                        reasons = []
                        if rom_m < min_rom:
                            reasons.append(f"ROM {rom_m*100:.0f}cm < {min_rom*100:.0f}cm")
                        if mcv < MIN_MCV_M_S:
                            reasons.append(f"MCV {mcv:.2f} < {MIN_MCV_M_S}")
                        print(f"  ❌ 拒绝: {'; '.join(reasons)}")
                state = "STANDING"
                y_lowest = -1
                y_bottom = -1
                frames_in_buffer = 0
                barbell_path_y_list = []
                current_rep_sequence = []

    cap.release()
    if return_seq:
        return results, (first_rep_sequence if first_rep_sequence else [])
    return results


def _resolve_single_video(video_arg, videos_dir):
    """将 --video 参数解析为绝对路径。支持：完整路径、或文件名（在 videos_dir 下模糊匹配）。"""
    if os.path.isfile(video_arg):
        return os.path.abspath(video_arg)
    base = os.path.basename(video_arg)
    if not os.path.isdir(videos_dir):
        return None
    for f in os.listdir(videos_dir):
        if not f.lower().endswith(('.mp4', '.mov', '.avi')):
            continue
        if f == base or f.startswith(base.split(".")[0]) or base.split(".")[0] in f:
            return os.path.join(videos_dir, f)
    return None


def main():
    parser = argparse.ArgumentParser(description="VBT 视频批处理 / 单视频补充分析")
    parser.add_argument("--video", type=str, default=None, metavar="FILE",
                        help="仅处理指定视频并补充到数据库（会先删除该文件已有记录）。例: --video IMG_1696 或 --video IMG_1696（1）.mov")
    args = parser.parse_args()

    videos_dir = VIDEOS_DIR
    if not os.path.isdir(videos_dir):
        os.makedirs(videos_dir, exist_ok=True)
        print(f"已创建目录 {videos_dir}，请放入 .mp4 / .mov / .avi 后重新运行。")
        return

    if args.video:
        video_path = _resolve_single_video(args.video.strip(), videos_dir)
        if not video_path:
            print(f"未找到视频: {args.video}（在 {videos_dir}/ 下无匹配的 .mp4/.mov/.avi 文件）。")
            return
        video_files = [video_path]
        print(f"单视频模式: {os.path.basename(video_path)}")
    else:
        video_files = []
        for f in os.listdir(videos_dir):
            if f.lower().endswith(('.mp4', '.mov', '.avi')):
                video_files.append(os.path.join(videos_dir, f))
        video_files = sorted(video_files)
        if not video_files:
            print(f"未在 {videos_dir}/ 下找到视频文件（.mp4 / .mov / .avi）。")
            return

    init_batch_table(DB_PATH)
    standard_seq = get_standard(DB_PATH)
    if standard_seq is None:
        print("未找到标准动作（standard_action 表为空），DTW 相似度将为空。可先运行 vbt_analytics_pro 录制标准动作。")

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    total_reps = 0
    for video_path in video_files:
        filename = os.path.basename(video_path)
        if args.video:
            deleted = delete_batch_reps_by_filename(DB_PATH, filename)
            if deleted:
                print(f"已删除该视频旧记录 {deleted} 条，重新分析并补充。")
        print(f"处理: {filename}")
        reps = process_video(video_path, interpreter, input_details, output_details, standard_seq)
        for rep_no, v_mean, min_knee_angle, max_trunk_angle, dtw_sim, barbell_path_blob in reps:
            insert_batch_rep(
                DB_PATH,
                filename=filename,
                rep_no=rep_no,
                v_mean=v_mean,
                min_knee_angle=min_knee_angle,
                max_trunk_angle=max_trunk_angle,
                dtw_similarity=dtw_sim,
                barbell_path_y=barbell_path_blob,
            )
            total_reps += 1
            ka = f"{min_knee_angle:.1f}°" if min_knee_angle is not None else "—"
            ta = f"{max_trunk_angle:.1f}°" if max_trunk_angle is not None else "—"
            sim = f"{dtw_sim:.3f}" if dtw_sim is not None else "—"
            print(f"  动作 {rep_no} | MCV={v_mean:.3f} m/s | 最小膝角={ka} | 躯干最大倾角={ta} | 相似度={sim}")

    print(f"批处理完成：共 {len(video_files)} 个视频，{total_reps} 条动作记录已写入 {DB_PATH} 表 {BATCH_TABLE}。")


if __name__ == "__main__":
    main()
