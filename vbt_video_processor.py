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
from datetime import datetime
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
    log_analysis_task,
)
from vbt_runtime_config import get_current_user_name, get_user_height_cm
from squat_analysis_core import (
    letterbox_preprocess,
    unpad_keypoints_array,
    CalibrationState,
    SquatStateMachine,
    HipTracker,
    MEDIAN_WINDOW,
    VALID_Y_HISTORY,
    MOVEMENT_THRESHOLD_RATIO,
    BUFFER_RATIO,
    FRAMES_IN_BUFFER_TO_END,
    MIN_ROM_RATIO,
    MIN_MCV_M_S,
)

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

VIDEOS_DIR = "videos"
BATCH_TABLE = "batch_reps"


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
    for col, typ in [("barbell_path_y", "BLOB"), ("velocity_loss", "REAL")]:
        try:
            cur.execute(f"ALTER TABLE batch_reps ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def insert_batch_rep(db_path, filename, rep_no, v_mean, min_knee_angle, max_trunk_angle, dtw_similarity, barbell_path_y=None, velocity_loss=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO batch_reps (filename, rep_no, v_mean, min_knee_angle, max_trunk_angle, dtw_similarity, barbell_path_y, velocity_loss)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (filename, rep_no, v_mean, min_knee_angle, max_trunk_angle, dtw_similarity, barbell_path_y, velocity_loss))
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


def process_video(video_path, interpreter, input_details, output_details, standard_seq=None, return_seq=False, user_height_cm=None):
    """
    统一离线批处理入口。
    与实时模式 (vbt_cv_engine.py) 使用完全相同的：
      - letterbox 预处理（消除几何失真）
      - unpad_keypoints_array 坐标还原
      - CalibrationState 标定状态机
      - SquatStateMachine / HipTracker 动作状态机
    物理时间严格基于帧索引 / 视频 FPS，严禁 time.time()。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    user_height_m = (float(user_height_cm or get_user_height_cm()) or 175.0) / 100.0

    calib = CalibrationState(user_height_m)
    tracker = HipTracker(is_bodyweight=False)
    state_machine: SquatStateMachine | None = None  # 标定完成后初始化

    frame_idx = 0
    results = []
    current_rep_sequence: list = []
    barbell_path_y_list: list = []
    first_rep_sequence = None
    min_left_knee = 180.0
    min_right_knee = 180.0
    max_trunk = -1.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # BGR -> RGB
        if frame.ndim == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w = frame_rgb.shape[:2]

        # ── 统一 letterbox 预处理（与实时模式完全相同） ──
        padded, offset_x, offset_y, scale = letterbox_preprocess(frame_rgb)
        inp = np.expand_dims(padded, axis=0).astype(np.uint8)
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        kps_raw = interpreter.get_tensor(output_details[0]["index"])[0][0]

        # ── 统一坐标还原（与实时模式完全相同） ──
        kps = unpad_keypoints_array(kps_raw, h, w, offset_x, offset_y, scale)

        # ── 标定 ──
        sm_state = state_machine.state if state_machine is not None else "STANDING"
        calib.update(kps, frame_idx, sm_state)
        if not calib.is_done:
            continue

        # 标定完成后延迟初始化状态机
        if state_machine is None:
            state_machine = SquatStateMachine(
                body_height_px=calib.body_height_px,
                user_height_m=user_height_m,
                video_fps=fps,
            )

        # ── 跟踪 ──
        smoothed_y, raw_y, raw_x = tracker.update(kps)
        if smoothed_y is None:
            continue

        # ── 角度计算（用于 results 输出） ──
        conf5  = float(kps[5][2])  if len(kps) > 5  else 0.0
        conf11 = float(kps[11][2]) if len(kps) > 11 else 0.0
        conf12 = float(kps[12][2]) if len(kps) > 12 else 0.0
        l_hip   = kps[11] if conf11 >= CONF_THRESHOLD else None
        r_hip   = kps[12] if conf12 >= CONF_THRESHOLD else None
        l_knee  = kps[13] if kps[13][2] > CONF_THRESHOLD else None
        r_knee  = kps[14] if kps[14][2] > CONF_THRESHOLD else None
        l_ankle = kps[15] if kps[15][2] > CONF_THRESHOLD else None
        r_ankle = kps[16] if kps[16][2] > CONF_THRESHOLD else None
        left_knee_deg  = angle_deg(l_hip, l_knee, l_ankle) if l_hip is not None else None
        right_knee_deg = angle_deg(r_hip, r_knee, r_ankle) if r_hip is not None else None
        shoulder_mid = (kps[5][0], kps[5][1]) if conf5 >= 0.2 else None
        hip_mid = (
            ((float(l_hip[0]) + float(r_hip[0])) / 2.0,
             (float(l_hip[1]) + float(r_hip[1])) / 2.0)
            if l_hip is not None and r_hip is not None else None
        )
        trunk_deg = trunk_angle_deg(shoulder_mid, hip_mid)
        lk = left_knee_deg  if left_knee_deg  is not None else 0.0
        rk = right_knee_deg if right_knee_deg is not None else 0.0
        tr = trunk_deg      if trunk_deg      is not None else 0.0

        # 累计本 rep 的角度/序列数据
        if state_machine.state in ("DOWN", "UP"):
            current_rep_sequence.append((lk, rk, tr))
            barbell_path_y_list.append(smoothed_y)
            if left_knee_deg  is not None: min_left_knee  = min(min_left_knee,  left_knee_deg)
            if right_knee_deg is not None: min_right_knee = min(min_right_knee, right_knee_deg)
            if trunk_deg      is not None: max_trunk       = max(max_trunk,      trunk_deg)
        elif state_machine.state == "STANDING":
            # 新 rep 开始时重置
            current_rep_sequence = []
            barbell_path_y_list  = []
            min_left_knee  = 180.0
            min_right_knee = 180.0
            max_trunk      = -1.0

        # ── 状态机推进 ──
        state_machine.update(smoothed_y, frame_idx, calib.ratio)

        # ── 消费完成的 rep ──
        if state_machine.finished_rep is not None:
            rep = state_machine.finished_rep
            state_machine.finished_rep = None

            min_knee_angle = min(min_left_knee, min_right_knee)
            if min_knee_angle > 179:
                min_knee_angle = None
            max_trunk_out = max_trunk if max_trunk >= 0 else None

            dtw_sim = None
            if standard_seq and current_rep_sequence:
                dtw_sim = dtw_similarity(current_rep_sequence, standard_seq)
                dtw_sim = round(dtw_sim, 4) if dtw_sim is not None else None

            path_blob = np.array(barbell_path_y_list, dtype=np.float32).tobytes() if barbell_path_y_list else None
            results.append((rep.rep_index, rep.mcv, min_knee_angle, max_trunk_out, dtw_sim, path_blob))

            if return_seq and rep.rep_index == 1 and len(current_rep_sequence) > 5:
                first_rep_sequence = list(current_rep_sequence)

            delta_y_m = rep.rom_m
            print(
                f"  ✅ 有效 rep {rep.rep_index} | MCV: {rep.mcv:.2f} m/s"
                f" | ROM: {delta_y_m*100:.0f}cm | t={rep.concentric_time_s:.2f}s"
            )

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
    parser.add_argument("--user-height", type=float, default=None, metavar="CM",
                        help="用户身高 (cm)，用于动态比例尺。不传则从 vbt_config.json 读取。")
    args = parser.parse_args()
    user_height_cm = args.user_height if args.user_height is not None else get_user_height_cm()

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
    total_videos = len(video_files)
    for idx, video_path in enumerate(video_files, start=1):
        filename = os.path.basename(video_path)
        if args.video:
            deleted = delete_batch_reps_by_filename(DB_PATH, filename)
            if deleted:
                print(f"已删除该视频旧记录 {deleted} 条，重新分析并补充。")
        print(f"正在分析第 {idx}/{total_videos} 个视频，请稍候... ({filename})")
        start_at = time.time()
        start_iso = datetime.now().isoformat()
        try:
            reps = list(process_video(video_path, interpreter, input_details, output_details, standard_seq, user_height_cm=user_height_cm))
            # 统一使用 Mean Velocity 计算 velocity_loss，严禁 peak_velocity / max_inst_vel
            rep_mean_velocities = [r[1] for r in reps]
            best_mean_vel = max(rep_mean_velocities) if rep_mean_velocities else 0.0
            for rep_no, v_mean, min_knee_angle, max_trunk_angle, dtw_sim, barbell_path_blob in reps:
                velocity_loss_pct = (
                    (best_mean_vel - v_mean) / best_mean_vel * 100.0
                    if best_mean_vel > 0 else 0.0
                )
                insert_batch_rep(
                    DB_PATH,
                    filename=filename,
                    rep_no=rep_no,
                    v_mean=v_mean,
                    min_knee_angle=min_knee_angle,
                    max_trunk_angle=max_trunk_angle,
                    dtw_similarity=dtw_sim,
                    barbell_path_y=barbell_path_blob,
                    velocity_loss=round(velocity_loss_pct, 1),
                )
                total_reps += 1
                ka = f"{min_knee_angle:.1f}°" if min_knee_angle is not None else "—"
                ta = f"{max_trunk_angle:.1f}°" if max_trunk_angle is not None else "—"
                sim = f"{dtw_sim:.3f}" if dtw_sim is not None else "—"
                vl = f"{velocity_loss_pct:.1f}%" if best_mean_vel > 0 else "—"
                print(f"  动作 {rep_no} | MCV={v_mean:.3f} m/s | 流失={vl} | 最小膝角={ka} | 躯干最大倾角={ta} | 相似度={sim}")
            # ── Finalize set: persist set summary + AI prediction ─────
            if rep_mean_velocities:
                try:
                    from vbt_set_finalizer import finalize_set
                    import uuid as _uuid
                    _vid_session_id = f"video_{os.path.splitext(filename)[0]}"
                    _rep_rows = [
                        {"v_mean": r[1], "rom": 0.3, "velocity_loss": 0.0,
                         "calib_is_fallback": False, "pose_issues": None,
                         "left_knee_angle": r[2] or 90.0,
                         "right_knee_angle": r[2] or 90.0,
                         "trunk_angle": r[3] or 15.0}
                        for r in reps
                    ]
                    _result = finalize_set(
                        session_id=_vid_session_id,
                        set_number=1,
                        user_name=get_current_user_name(),
                        mode="Strength",
                        load_kg=0.0,
                        rep_rows=_rep_rows,
                        db_path=DB_PATH,
                    )
                    if _result:
                        print(f"  AI 推荐: {_result.recommendation_action} (fatigue={_result.fatigue_risk:.2f}, fallback={_result.fatigue_model_status})")
                except Exception as _fe:
                    print(f"  [警告] set finalize 失败: {_fe}")
            log_analysis_task(
                DB_PATH,
                video_name=filename,
                user_name=get_current_user_name(),
                start_time=start_iso,
                duration=time.time() - start_at,
                reps_count=len(reps),
                status="Success",
                error_msg=None,
            )
        except Exception as e:
            log_analysis_task(
                DB_PATH,
                video_name=filename,
                user_name=get_current_user_name(),
                start_time=start_iso,
                duration=time.time() - start_at,
                reps_count=0,
                status="Failed",
                error_msg=str(e),
            )
            raise

    print(f"批处理完成：共 {len(video_files)} 个视频，{total_reps} 条动作记录已写入 {DB_PATH} 表 {BATCH_TABLE}。")


if __name__ == "__main__":
    main()
