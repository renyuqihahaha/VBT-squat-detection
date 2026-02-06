#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBT 增强版分析系统 vbt_analytics_pro.py
- 多源输入：OpenCV 本地 .mp4 离线批处理 / 摄像头实时
- 生物力学：双侧膝角(12-14-16 / 11-13-15)、躯干倾角 [cite: 2026-01-25]
- DTW 与标准动作相似度 [cite: 2026-01-25]
- sqlite3 持久化、matplotlib 三位一体报表
"""

import argparse
import cv2
import numpy as np
import time
import sqlite3
import os
from collections import deque
from datetime import datetime

# 可选：摄像头模式才需要
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ================= 核心参数配置 [cite: 2026-01-24, 2026-01-25] =================
MODEL_PATH = "models/movenet_lightning.tflite"
AWB_MODE = 4
SWAP_CHANNELS = False
CONF_THRESHOLD = 0.25
DEFAULT_SCALE = 0.0031   # 1px ≈ 3mm
TRAJECTORY_LEN = 40
DB_PATH = "squat_gym.db"
# =========================================================================

# MoveNet 关节点索引: 5,6 肩 9,10 腕 11,12 髋 13,14 膝 15,16 踝
# 左膝角: 11-13-15  右膝角: 12-14-16  [cite: 2026-01-25]


def angle_deg(p1, p2, p3):
    """三点夹角（度），p2 为顶点。p 为 (y, x) 或 (x, y)，统一按 (x,y) 向量处理。"""
    if p1 is None or p2 is None or p3 is None:
        return None
    # MoveNet 输出 keypoints[i] = (y, x, conf)，取 (x, y) 做平面角
    def vec(a, b):
        return np.array([a[1] - b[1], a[0] - b[0]], dtype=float)
    v1 = vec(p1, p2)
    v2 = vec(p3, p2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def trunk_angle_deg(shoulder_mid, hip_mid):
    """
    躯干倾角：肩部中点与髋部中点的连线与垂直线的夹角 [cite: 2026-01-25]。
    垂直向上为 0°，前倾为正（度）。
    """
    if shoulder_mid is None or hip_mid is None:
        return None
    # 肩-髋向量 (dx, dy)，图像坐标系 y 向下
    dx = shoulder_mid[1] - hip_mid[1]  # x 方向
    dy = shoulder_mid[0] - hip_mid[0]  # y 方向（向上为负）
    # 垂直线向上单位向量 (0, -1) 在 (x,y)
    # 躯干向量 (dx, dy)，与垂直夹角: cos = (dx*0 + dy*(-1)) / |v| = -dy/|v|
    norm = np.sqrt(dx*dx + dy*dy)
    if norm < 1e-6:
        return None
    cos_a = np.clip(-dy / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def dtw_distance(seq_a, seq_b):
    """
    一维序列 DTW 距离（欧氏距离）。seq_a, seq_b 为等长特征的列表/数组。
    若为多特征，将每帧拼成单维或对多列分别 DTW 后加权，此处用简单平均多列。
    """
    a = np.asarray(seq_a)
    b = np.asarray(seq_b)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float('inf')
    # 多列时按列平均成单列再 DTW
    if a.shape[1] > 1:
        a = np.mean(a, axis=1, keepdims=True)
    if b.shape[1] > 1:
        b = np.mean(b, axis=1, keepdims=True)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.abs(a[i-1, 0] - b[j-1, 0])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return D[n, m]


def dtw_similarity(seq_current, seq_standard):
    """
    基于 DTW 的相似度评分 [cite: 2026-01-25]。
    相似度 = 1 / (1 + normalized_dtw)，范围约 (0, 1]，越大越相似。
    """
    if not seq_standard or not seq_current:
        return None
    d = dtw_distance(seq_current, seq_standard)
    # 归一化：除以较长序列长度，避免长度主导
    norm = max(len(seq_current), len(seq_standard), 1)
    scale = np.mean(np.asarray(seq_standard)) if np.asarray(seq_standard).size else 1.0
    if abs(scale) < 1e-6:
        scale = 1.0
    normalized = (d / norm) / (abs(scale) + 1e-6)
    return 1.0 / (1.0 + normalized)


# --------------- 数据库 ---------------
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            rep_count INTEGER NOT NULL,
            v_mean REAL,
            rom REAL,
            left_knee_angle REAL,
            right_knee_angle REAL,
            trunk_angle REAL,
            dtw_similarity REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS standard_action (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_idx INTEGER NOT NULL,
            left_knee REAL,
            right_knee REAL,
            trunk REAL
        )
    """)
    conn.commit()
    conn.close()


def insert_rep(db_path, rep_count, v_mean, rom, left_knee, right_knee, trunk, dtw_sim):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO reps (ts, rep_count, v_mean, rom, left_knee_angle, right_knee_angle, trunk_angle, dtw_similarity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), rep_count, v_mean, rom, left_knee, right_knee, trunk, dtw_sim))
    conn.commit()
    conn.close()


def get_standard(db_path):
    """支持两种表结构：blob 格式 (action_name, sequence_data) 或行格式 (frame_idx, left_knee, right_knee, trunk)。"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(standard_action)")
    cols = [row[1] for row in cur.fetchall()]
    if "sequence_data" in cols:
        cur.execute("SELECT sequence_data FROM standard_action LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if not row or row[0] is None:
            return None
        arr = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 3)
        return [tuple(r) for r in arr]
    cur.execute("SELECT frame_idx, left_knee, right_knee, trunk FROM standard_action ORDER BY frame_idx")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return None
    return [(r[1], r[2], r[3]) for r in rows]


def save_standard(db_path, sequence):
    """sequence: list of (left_knee, right_knee, trunk)。支持 blob 表或行表。"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(standard_action)")
    cols = [row[1] for row in cur.fetchall()]
    if "sequence_data" in cols:
        cur.execute("DELETE FROM standard_action")
        blob = np.array(sequence, dtype=np.float32).tobytes()
        cur.execute("INSERT INTO standard_action (action_name, sequence_data) VALUES (?, ?)",
                    ("Gold_Standard_Squat", blob))
    else:
        cur.execute("DELETE FROM standard_action")
        for i, (lk, rk, tr) in enumerate(sequence):
            cur.execute("INSERT INTO standard_action (frame_idx, left_knee, right_knee, trunk) VALUES (?, ?, ?, ?)",
                        (i, lk or 0, rk or 0, tr or 0))
    conn.commit()
    conn.close()


# --------------- 视频源 ---------------
def create_video_source(source_arg):
    """多源输入：摄像头或本地 .mp4 [cite: 2026-01-25]。"""
    if source_arg is None or source_arg.strip().lower() == "camera":
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
            picam2.configure(config)
            picam2.start()
            picam2.set_controls({"AwbMode": AWB_MODE})
            return ("camera", picam2, None)
        except Exception as e:
            print("摄像头不可用，请使用 --source video.mp4")
            raise e
    path = source_arg.strip()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"视频文件不存在: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    return ("video", None, cap)


def get_frame(source_type, picam2, cap):
    if source_type == "camera":
        frame = picam2.capture_array()
        return frame
    ret, frame = cap.read()
    if not ret:
        return None
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def release_source(source_type, picam2, cap):
    if source_type == "camera" and picam2 is not None:
        picam2.stop()
    if cap is not None:
        cap.release()


# --------------- 主流程 ---------------
def run_analytics(source_arg=None):
    parser = argparse.ArgumentParser(description="VBT 增强版分析：多源输入 / 生物力学 / DTW / 持久化 / 报表")
    parser.add_argument("--source", default=source_arg, help="camera 或本地 .mp4 路径")
    parser.add_argument("--no-plot", action="store_true", help="结束时不再生成本次训练曲线图")
    args = parser.parse_args()
    source_arg = args.source or "camera"

    init_db(DB_PATH)
    source_type, picam2, cap = create_video_source(source_arg)
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 状态机与指标 [cite: 2026-01-24, 2026-01-25]
    state = "GESTURE_WAIT"
    rep_count = 0
    v_mean, rom = 0.0, 0.0
    trajectory = deque(maxlen=TRAJECTORY_LEN)
    y_base, y_lowest = 0, 999
    t_start_asc = 0

    # 当前 rep 时间序列（用于 DTW）
    current_rep_sequence = []
    # 本次训练所有 rep 的指标（用于报表）
    session_records = []

    standard_seq = get_standard(DB_PATH)
    if standard_seq is not None:
        print("已加载标准动作序列，将计算 DTW 相似度。")

    print("VBT 增强版分析已启动。输入: " + ("摄像头" if source_type == "camera" else source_arg))
    if source_type == "camera":
        print("请举起双手开启追踪。")

    try:
        while True:
            frame = get_frame(source_type, picam2, cap)
            if frame is None:
                break
            display_img = frame.copy() if not SWAP_CHANNELS else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w, _ = display_img.shape

            input_data = np.expand_dims(cv2.resize(frame, (192, 192)), axis=0).astype(np.uint8)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]

            def get_kp(i):
                return keypoints[i] if keypoints[i][2] > CONF_THRESHOLD else None

            l_shld, r_shld = get_kp(5), get_kp(6)
            l_wris, r_wris = get_kp(9), get_kp(10)
            l_hip, r_hip = get_kp(11), get_kp(12)
            l_knee, r_knee = get_kp(13), get_kp(14)
            l_ankle, r_ankle = get_kp(15), get_kp(16)

            # 生物力学：双侧膝角 11-13-15 / 12-14-16，躯干倾角 [cite: 2026-01-25]
            left_knee_deg = angle_deg(l_hip, l_knee, l_ankle)
            right_knee_deg = angle_deg(r_hip, r_knee, r_ankle)
            shoulder_mid = ((l_shld[0]+r_shld[0])/2, (l_shld[1]+r_shld[1])/2) if (l_shld and r_shld) else None
            hip_mid = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2) if (l_hip and r_hip) else None
            trunk_deg = trunk_angle_deg(shoulder_mid, hip_mid)

            # 当前帧角度序列（用于 DTW）
            lk = left_knee_deg if left_knee_deg is not None else 0.0
            rk = right_knee_deg if right_knee_deg is not None else 0.0
            tr = trunk_deg if trunk_deg is not None else 0.0

            # --- A. 手势启动 ---
            if state == "GESTURE_WAIT":
                if all(pt is not None for pt in [l_shld, r_shld, l_wris, r_wris]):
                    if l_wris[0] < l_shld[0] and r_wris[0] < r_shld[0]:
                        hips = [p for p in [l_hip, r_hip] if p is not None]
                        if hips:
                            y_base = sum(p[0] for p in hips) / len(hips) * h
                            state = "STANDING"
                            rep_count = 0
                            trajectory.clear()
                            current_rep_sequence.clear()
                            print("监测已激活！")

            # --- B. 深蹲状态机 ---
            elif state != "GESTURE_WAIT":
                hips = [p for p in [l_hip, r_hip] if p is not None]
                if hips:
                    curr_y_px = sum(p[0] for p in hips) / len(hips) * h
                    curr_x_px = sum(p[1] for p in hips) / len(hips) * w
                    trajectory.appendleft((int(curr_x_px), int(curr_y_px)))
                    # 仅在 DOWN/UP 阶段收集当前 rep 序列（用于 DTW）
                    if state in ("DOWN", "UP"):
                        current_rep_sequence.append((lk, rk, tr))

                    if state == "STANDING":
                        if curr_y_px > y_base + 20:
                            state = "DOWN"
                            y_lowest = curr_y_px
                            current_rep_sequence.clear()
                            current_rep_sequence.append((lk, rk, tr))

                    elif state == "DOWN":
                        if curr_y_px > y_lowest:
                            y_lowest = curr_y_px
                        if curr_y_px < y_lowest - 15:
                            state = "UP"
                            t_start_asc = time.time()

                    elif state == "UP":
                        if curr_y_px < y_base + 15:
                            duration = time.time() - t_start_asc
                            if duration > 0.1:
                                rom = (y_lowest - curr_y_px) * DEFAULT_SCALE
                                v_mean = rom / duration
                                rep_count += 1
                                # DTW 相似度 [cite: 2026-01-25]
                                dtw_sim = None
                                if standard_seq and current_rep_sequence:
                                    dtw_sim = dtw_similarity(current_rep_sequence, standard_seq)
                                    dtw_sim = round(dtw_sim, 4) if dtw_sim is not None else None
                                else:
                                    # 首个 rep 可作为标准动作
                                    if get_standard(DB_PATH) is None and len(current_rep_sequence) > 5:
                                        save_standard(DB_PATH, current_rep_sequence)
                                        standard_seq = current_rep_sequence
                                        dtw_sim = 1.0
                                        print("已保存为首个标准动作。")

                                lk_f = left_knee_deg
                                rk_f = right_knee_deg
                                tr_f = trunk_deg
                                # 数据持久化：PHASE UP 结束时写入 [cite: 2026-01-25]
                                insert_rep(DB_PATH, rep_count, v_mean, rom,
                                           lk_f, rk_f, tr_f, dtw_sim)
                                session_records.append({
                                    "rep": rep_count,
                                    "v_mean": v_mean,
                                    "rom": rom,
                                    "left_knee": lk_f,
                                    "right_knee": rk_f,
                                    "trunk": tr_f,
                                    "dtw_similarity": dtw_sim,
                                })
                                if dtw_sim is not None:
                                    print(f"Rep {rep_count} | V_mean: {v_mean:.2f} m/s | ROM: {rom:.2f} m | DTW相似度: {dtw_sim:.3f}")
                            state = "STANDING"
                            y_lowest = 999
                            current_rep_sequence.clear()

            # --- C. UI ---
            for i in range(1, len(trajectory)):
                cv2.line(display_img, trajectory[i-1], trajectory[i], (0, 255, 0), 2)
            overlay = display_img.copy()
            cv2.rectangle(overlay, (10, 10), (320, 220), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display_img, 0.5, 0, display_img)

            cv2.putText(display_img, f"REPS: {rep_count}", (25, 50), 2, 1.1, (0, 255, 255), 2)
            cv2.putText(display_img, f"V_mean: {v_mean:.2f} m/s", (25, 90), 2, 0.9, (255, 255, 0), 2)
            cv2.putText(display_img, f"ROM: {rom:.2f} m", (25, 130), 2, 0.9, (0, 255, 0), 2)
            cv2.putText(display_img, f"L/R knee: {left_knee_deg or 0:.0f} / {right_knee_deg or 0:.0f}", (25, 170), 2, 0.65, (200, 200, 255), 2)
            cv2.putText(display_img, f"Trunk: {trunk_deg or 0:.0f}", (25, 205), 2, 0.65, (200, 200, 255), 2)

            status_color = (0, 0, 255) if state == "GESTURE_WAIT" else (0, 255, 0)
            status_text = "READY: RAISE HANDS" if state == "GESTURE_WAIT" else f"PHASE: {state}"
            cv2.putText(display_img, status_text, (w - 260, 40), 1, 1.2, status_color, 2)

            cv2.imshow("VBT Analytics Pro", display_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        release_source(source_type, picam2, cap)
        cv2.destroyAllWindows()

    # 可视化报表：速度/位移/角度三位一体 [cite: 2026-01-25]
    if not args.no_plot and session_records:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            reps = [r["rep"] for r in session_records]
            v_means = [r["v_mean"] for r in session_records]
            roms = [r["rom"] for r in session_records]
            left_knees = [r["left_knee"] or 0 for r in session_records]
            right_knees = [r["right_knee"] or 0 for r in session_records]
            trunks = [r["trunk"] or 0 for r in session_records]

            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(reps, v_means, "o-", color="C0", label="平均速度 (m/s)")
            axes[0].set_ylabel("速度 (m/s)")
            axes[0].legend(loc="upper right")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(reps, roms, "s-", color="C1", label="ROM (m)")
            axes[1].set_ylabel("位移 ROM (m)")
            axes[1].legend(loc="upper right")
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(reps, left_knees, "o-", label="左膝角 (°)")
            axes[2].plot(reps, right_knees, "s-", label="右膝角 (°)")
            axes[2].plot(reps, trunks, "^-", label="躯干倾角 (°)")
            axes[2].set_xlabel("Rep 次数")
            axes[2].set_ylabel("角度 (°)")
            axes[2].legend(loc="upper right")
            axes[2].grid(True, alpha=0.3)

            plt.suptitle("本次训练：速度 / 位移 / 角度")
            plt.tight_layout()
            out_path = "vbt_analytics_report.png"
            plt.savefig(out_path, dpi=120)
            plt.close()
            print(f"报表已保存: {out_path}")
        except Exception as e:
            print("生成报表失败:", e)

    return session_records


if __name__ == "__main__":
    run_analytics()
