import cv2
import numpy as np
import time
import csv
from collections import deque
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2

# ================= 核心参数配置 [cite: 2026-01-24, 2026-01-25] =================
MODEL_PATH = "models/movenet_lightning.tflite"
AWB_MODE = 4            # 已验证的日光模式
SWAP_CHANNELS = False    # 已验证的 BGR 模式
CONF_THRESHOLD = 0.25   
DEFAULT_SCALE = 0.0031  # 默认比例系数 (1px ≈ 3mm)
TRAJECTORY_LEN = 40     # 轨迹线保留长度
# =========================================================================

# 1. 引擎与硬件初始化
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()
picam2.set_controls({"AwbMode": AWB_MODE})

# 2. 状态机与指标变量 [cite: 2026-01-24, 2026-01-25]
state = "GESTURE_WAIT" # GESTURE_WAIT, STANDING, DOWN, UP
rep_count = 0
v_mean, rom = 0.0, 0.0
trajectory = deque(maxlen=TRAJECTORY_LEN)

y_base, y_lowest = 0, 999
t_start_asc = 0

print("🚀 完美深蹲 VBT 系统已启动！请举起双手开启追踪。")

try:
    while True:
        frame = picam2.capture_array()
        # 图像色彩校正
        display_img = frame.copy() if not SWAP_CHANNELS else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w, _ = display_img.shape

        # AI 推理 (MoveNet)
        input_data = np.expand_dims(cv2.resize(frame, (192, 192)), axis=0).astype(np.uint8)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]

        # 提取核心关节点 [cite: 2026-01-25]
        def get_kp(i): return keypoints[i] if keypoints[i][2] > CONF_THRESHOLD else None
        l_shld, r_shld = get_kp(5), get_kp(6)
        l_wris, r_wris = get_kp(9), get_kp(10)
        l_hip, r_hip = get_kp(11), get_kp(12)

        # --- A. 手势启动逻辑 [cite: 2026-01-25] ---
        if state == "GESTURE_WAIT":
            if all(pt is not None for pt in [l_shld, r_shld, l_wris, r_wris]):
                if l_wris[0] < l_shld[0] and r_wris[0] < r_shld[0]: # 手比肩高
                    hips = [p for p in [l_hip, r_hip] if p is not None]
                    if hips:
                        y_base = sum(p[0] for p in hips) / len(hips) * h
                        state = "STANDING"; rep_count = 0; trajectory.clear()
                        print("🔔 监测已激活！")

        # --- B. 完美深蹲状态机 (重点识别髋关节) [cite: 2026-01-24] ---
        elif state != "GESTURE_WAIT":
            hips = [p for p in [l_hip, r_hip] if p is not None]
            if hips:
                curr_y_px = sum(p[0] for p in hips) / len(hips) * h
                curr_x_px = sum(p[1] for p in hips) / len(hips) * w
                trajectory.appendleft((int(curr_x_px), int(curr_y_px)))

                if state == "STANDING":
                    if curr_y_px > y_base + 20: # 离心阶段开始
                        state = "DOWN"; y_lowest = curr_y_px
                
                elif state == "DOWN":
                    if curr_y_px > y_lowest: y_lowest = curr_y_px
                    if curr_y_px < y_lowest - 15: # 向心阶段开始
                        state = "UP"; t_start_asc = time.time()
                
                elif state == "UP":
                    if curr_y_px < y_base + 15: # 动作完成
                        duration = time.time() - t_start_asc
                        if duration > 0.1:
                            rom = (y_lowest - curr_y_px) * DEFAULT_SCALE
                            v_mean = rom / duration # 计算平均向心速度
                            rep_count += 1
                        state = "STANDING"; y_lowest = 999

        # --- C. UI 渲染 (仪表盘与轨迹) [cite: 2026-01-24] ---
        # 1. 绘制运动轨迹
        for i in range(1, len(trajectory)):
            cv2.line(display_img, trajectory[i-1], trajectory[i], (0, 255, 0), 2)
        
        # 2. 绘制左侧半透明面板
        overlay = display_img.copy()
        cv2.rectangle(overlay, (10, 10), (300, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display_img, 0.5, 0, display_img)
        
        cv2.putText(display_img, f"REPS: {rep_count}", (25, 50), 2, 1.1, (0, 255, 255), 2)
        cv2.putText(display_img, f"V_mean: {v_mean:.2f} m/s", (25, 100), 2, 0.9, (255, 255, 0), 2)
        cv2.putText(display_img, f"ROM: {rom:.2f} m", (25, 145), 2, 0.9, (0, 255, 0), 2)
        
        # 状态提示
        status_color = (0, 0, 255) if state == "GESTURE_WAIT" else (0, 255, 0)
        status_text = "READY: RAISE HANDS" if state == "GESTURE_WAIT" else f"PHASE: {state}"
        cv2.putText(display_img, status_text, (w-260, 40), 1, 1.2, status_color, 2)

        cv2.imshow('Perfect VBT System', display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    picam2.stop()
    cv2.destroyAllWindows()