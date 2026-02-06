#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强制提取版基准录入：不依赖“计数逻辑”，直接把视频里检测到的所有运动数据作为标准曲线 [cite: 2026-01-24]。
"""

import cv2
import sqlite3
import numpy as np
import os

# 引用你的工具库
from vbt_analytics_pro import angle_deg, trunk_angle_deg, init_db

# 配置路径
TARGET_VIDEO = "/home/kiki-pi/vbt_project/videos/dji_export_20260128_172531_1769592331891_compose_0.MOV"
DB_NAME = "squat_gym.db"
MODEL_PATH = "models/movenet_lightning.tflite"
CONF_THRESHOLD = 0.2
INPUT_SIZE = 192

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter


def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]["index"])
    return keypoints


def save_standard_forced():
    # 1. 初始化模型
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(TARGET_VIDEO)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {TARGET_VIDEO}")
        return

    full_sequence = []
    frame_count = 0
    print(f"🚀 开始强制分析: {os.path.basename(TARGET_VIDEO)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 2. 图像预处理 (Resize & 与 vbt 一致用 uint8)
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(img, axis=0).astype(np.uint8)

        # 3. 推理
        keypoints_with_scores = run_inference(interpreter, input_image)
        keypoints = keypoints_with_scores[0][0]  # [17, 3]

        # 4. 提取关键点：左 11-13-15，右 12-14-16；肩 5,6
        l_hip = keypoints[11]
        l_knee = keypoints[13]
        l_ankle = keypoints[15]
        r_hip = keypoints[12]
        r_knee = keypoints[14]
        r_ankle = keypoints[16]
        l_shld = keypoints[5]
        r_shld = keypoints[6]

        # 置信度检查 (>0.2 才参与计算)
        lk = None
        rk = None
        if l_hip[2] > CONF_THRESHOLD and l_knee[2] > CONF_THRESHOLD and l_ankle[2] > CONF_THRESHOLD:
            lk = angle_deg(l_hip, l_knee, l_ankle)
        if r_hip[2] > CONF_THRESHOLD and r_knee[2] > CONF_THRESHOLD and r_ankle[2] > CONF_THRESHOLD:
            rk = angle_deg(r_hip, r_knee, r_ankle)

        shoulder_mid = None
        hip_mid = None
        if l_shld[2] > CONF_THRESHOLD and r_shld[2] > CONF_THRESHOLD:
            shoulder_mid = ((l_shld[0] + r_shld[0]) / 2, (l_shld[1] + r_shld[1]) / 2)
        if l_hip[2] > CONF_THRESHOLD and r_hip[2] > CONF_THRESHOLD:
            hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        tr = trunk_angle_deg(shoulder_mid, hip_mid) if (shoulder_mid and hip_mid) else None

        # 每帧记录 (左膝角, 右膝角, 躯干角)，与 get_standard/DTW 格式一致
        lk_f = float(lk) if lk is not None else 0.0
        rk_f = float(rk) if rk is not None else 0.0
        tr_f = float(tr) if tr is not None else 0.0
        full_sequence.append((lk_f, rk_f, tr_f))

        if frame_count % 50 == 0:
            print(f"   Frame {frame_count}: 左膝 {lk_f:.0f}° 右膝 {rk_f:.0f}° 躯干 {tr_f:.0f}°")

    cap.release()

    # 5. 结果检查与入库
    if len(full_sequence) < 10:
        print(f"❌ 提取失败：仅提取到 {len(full_sequence)} 帧有效数据。请检查视频光线或是否拍到了全身。")
        return

    print(f"✅ 提取完成！共获取 {len(full_sequence)} 帧骨架数据。")

    # 存入数据库
    init_db(DB_NAME)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS standard_action")
    c.execute("""CREATE TABLE standard_action
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  action_name TEXT,
                  sequence_data BLOB,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    seq_blob = np.array(full_sequence, dtype=np.float32).tobytes()
    c.execute("INSERT INTO standard_action (action_name, sequence_data) VALUES (?, ?)",
              ("My_Gold_Standard", seq_blob))

    conn.commit()
    conn.close()
    print("💾 标准动作已成功保存到数据库！")


if __name__ == "__main__":
    save_standard_forced()
