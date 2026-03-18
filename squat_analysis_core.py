#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
squat_analysis_core.py — 深蹲分析统一核心

实时模式 (vbt_cv_engine.py) 与离线批处理模式 (vbt_video_processor.py) 共用此模块。

提供：
  1. letterbox_preprocess()   — 保持宽高比缩放+填充，避免几何失真
  2. unpad_keypoint()         — 还原归一化坐标为原图像素坐标
  3. unpad_keypoints_array()  — 批量版
  4. CalibrationState         — 标定状态机（肩踝/头踝/肩髋三级降级）
  5. SquatStateMachine        — 深蹲动作状态机（STANDING/DOWN/UP）
  6. compute_rep_metrics()    — 给定一次完整 rep，计算 MCV/ROM/velocity_loss

约束：
  - 不依赖 OpenCV 以外的图形库
  - 不含任何数据库写入逻辑
  - 物理时间严格基于帧索引 / 视频 FPS，不使用 time.time()
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("squat_analysis_core")

# ──────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────
MODEL_INPUT_SIZE = 192

ANATOMY_RATIO_SHOULDER_ANKLE = 0.80
ANATOMY_RATIO_HEAD_ANKLE = 0.90
ANATOMY_RATIO_SHOULDER_HIP = 0.30

MIN_CALIB_SAMPLES = 10
MIN_CALIB_SAMPLES_TERTIARY = 20
CALIB_VISIBILITY_RELAXED = 0.3
CALIB_FREEZE_FRAMES = 30
CALIB_FORCE_FRAMES = 30
CALIB_TIMEOUT_FRAMES = 50
DEFAULT_FALLBACK_RATIO = 0.004
ANKLE_FALLBACK_FRAMES = 5

MOVEMENT_THRESHOLD_RATIO = 0.08
BUFFER_RATIO = 0.05
FRAMES_IN_BUFFER_TO_END = 5
MIN_ROM_RATIO = 0.10
MIN_MCV_M_S = 0.02
MEDIAN_WINDOW = 9
VALID_Y_HISTORY = 5

DEPTH_MIN_M = 0.40
DEPTH_MAX_M = 0.70


# ──────────────────────────────────────────────────────────────
# 1. Letterbox 预处理
# ──────────────────────────────────────────────────────────────

def letterbox_preprocess(
    frame_rgb: np.ndarray,
    target: int = MODEL_INPUT_SIZE,
) -> tuple:
    """
    保持宽高比缩放 + 黑边填充至 target x target。
    Returns: (padded_image, offset_x, offset_y, scale)
    """
    h, w = frame_rgb.shape[:2]
    scale = target / max(h, w)
    w_new = int(w * scale)
    h_new = int(h * scale)
    resized = cv2.resize(frame_rgb, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    pad_w = target - w_new
    pad_h = target - h_new
    offset_x = pad_w / 2.0
    offset_y = pad_h / 2.0
    top = int(pad_h // 2)
    bottom = int(pad_h - pad_h // 2)
    left = int(pad_w // 2)
    right = int(pad_w - pad_w // 2)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    return padded, offset_x, offset_y, scale


# ──────────────────────────────────────────────────────────────
# 2. 坐标还原
# ──────────────────────────────────────────────────────────────

def unpad_keypoint(
    y_norm: float, x_norm: float,
    h: int, w: int,
    offset_x: float, offset_y: float,
    scale: float,
) -> tuple:
    """
    将 MoveNet 归一化坐标 [0,1] 还原为原图像素坐标 (y_px, x_px)。
    """
    y_canvas = y_norm * MODEL_INPUT_SIZE
    x_canvas = x_norm * MODEL_INPUT_SIZE
    y_content = y_canvas - offset_y
    x_content = x_canvas - offset_x
    h_new = h * scale
    w_new = w * scale
    y_orig = y_content / h_new * h if h_new > 0 else 0.0
    x_orig = x_content / w_new * w if w_new > 0 else 0.0
    return float(y_orig), float(x_orig)


def unpad_keypoints_array(
    kps_raw: np.ndarray,
    h: int, w: int,
    offset_x: float, offset_y: float,
    scale: float,
) -> np.ndarray:
    """
    批量还原关键点坐标。
    kps_raw: shape (17, 3), [y_norm, x_norm, conf]
    Returns: shape (17, 3), [y_px, x_px, conf]
    """
    out = np.zeros_like(kps_raw)
    for i in range(min(17, len(kps_raw))):
        y_px, x_px = unpad_keypoint(
            float(kps_raw[i][0]), float(kps_raw[i][1]),
            h, w, offset_x, offset_y, scale,
        )
        out[i][0] = y_px
        out[i][1] = x_px
        out[i][2] = kps_raw[i][2]
    return out


# ──────────────────────────────────────────────────────────────
# 3. 标定状态机
# ──────────────────────────────────────────────────────────────

class CalibrationState:
    """
    动态解剖自适应标定。
    优先级: 肩踝法 > 头踝法 > 肩髋法 > 超时兜底。
    调用方每帧执行 update(kps_pixel, frame_n, state)，
    kps_pixel 须已经过 unpad_keypoints_array 还原为原图像素坐标。
    """

    def __init__(self, user_height_m: float) -> None:
        self.user_height_m = float(user_height_m)
        self._samples_primary: list = []    # 肩踝
        self._samples_secondary: list = []  # 头踝
        self._samples_tertiary: list = []   # 肩髋
        self._force_tertiary = False
        self.is_done = False
        self.is_fallback = False
        self.ratio: float = 0.0
        self.body_height_px: float = 0.0
        self.method: str = "none"

    def _try_commit(self) -> bool:
        h_m = self.user_height_m
        if not self._force_tertiary and len(self._samples_primary) >= MIN_CALIB_SAMPLES:
            med = float(np.median(self._samples_primary))
            self.ratio = (h_m * ANATOMY_RATIO_SHOULDER_ANKLE) / med
            self.body_height_px = med / ANATOMY_RATIO_SHOULDER_ANKLE
            self.is_done = True
            self.method = "shoulder_ankle"
            logger.info("标定(肩踝法): median_px=%.1f ratio=%.6f m/px", med, self.ratio)
            return True
        if not self._force_tertiary and len(self._samples_secondary) >= MIN_CALIB_SAMPLES:
            med = float(np.median(self._samples_secondary))
            self.ratio = (h_m * ANATOMY_RATIO_HEAD_ANKLE) / med
            self.body_height_px = med / ANATOMY_RATIO_HEAD_ANKLE
            self.is_done = True
            self.method = "head_ankle"
            logger.info("标定(头踝法): median_px=%.1f ratio=%.6f m/px", med, self.ratio)
            return True
        if len(self._samples_tertiary) >= MIN_CALIB_SAMPLES_TERTIARY:
            med = float(np.median(self._samples_tertiary))
            self.ratio = (h_m * ANATOMY_RATIO_SHOULDER_HIP) / med
            self.body_height_px = med / ANATOMY_RATIO_SHOULDER_HIP
            self.is_done = True
            self.is_fallback = True
            self.method = "shoulder_hip"
            logger.warning("标定(躯干兜底): median_px=%.1f ratio=%.6f m/px", med, self.ratio)
            return True
        return False

    def update(self, kps: np.ndarray, frame_n: int, state: str) -> None:
        """
        每帧调用。kps: shape (17,3), 原图像素坐标。
        state: 'STANDING' / 'DOWN' / 'UP'。
        """
        if self.is_done:
            return

        vis = CALIB_VISIBILITY_RELAXED
        in_window = frame_n <= max(CALIB_FREEZE_FRAMES, CALIB_FORCE_FRAMES)
        allow_sample = in_window and (state == "STANDING" or frame_n <= CALIB_FORCE_FRAMES)

        # 超时兜底
        if frame_n > CALIB_TIMEOUT_FRAMES:
            self.ratio = DEFAULT_FALLBACK_RATIO
            self.body_height_px = self.user_height_m / DEFAULT_FALLBACK_RATIO
            self.is_done = True
            self.is_fallback = True
            self.method = "timeout"
            logger.warning("标定超时(frame %d)，使用兜底 %.6f m/px", frame_n, DEFAULT_FALLBACK_RATIO)
            return

        if not allow_sample:
            self._try_commit()
            return

        conf5  = float(kps[5][2])  if len(kps) > 5  else 0.0
        conf6  = float(kps[6][2])  if len(kps) > 6  else 0.0
        conf0  = float(kps[0][2])  if len(kps) > 0  else 0.0
        conf11 = float(kps[11][2]) if len(kps) > 11 else 0.0
        conf12 = float(kps[12][2]) if len(kps) > 12 else 0.0
        la_conf = float(kps[15][2]) if len(kps) > 15 else 0.0
        ra_conf = float(kps[16][2]) if len(kps) > 16 else 0.0

        shoulders_ok = conf5 >= vis or conf6 >= vis
        ankles_ok    = la_conf >= vis or ra_conf >= vis
        nose_ok      = conf0 >= vis
        hips_ok      = conf11 >= vis or conf12 >= vis

        # 前 ANKLE_FALLBACK_FRAMES 帧无踝 => 强制肩髋
        if frame_n <= ANKLE_FALLBACK_FRAMES and not ankles_ok and hips_ok:
            self._force_tertiary = True

        # 采集样本
        if not self._force_tertiary and shoulders_ok and ankles_ok:
            shoulder_ys = [float(kps[i][0]) for i in (5, 6) if float(kps[i][2]) >= vis]
            ankle_ys    = [float(kps[i][0]) for i in (15, 16) if float(kps[i][2]) >= vis]
            if shoulder_ys and ankle_ys:
                dist = abs(float(np.mean(ankle_ys)) - float(np.mean(shoulder_ys)))
                if dist > 30:
                    self._samples_primary.append(dist)
        elif not self._force_tertiary and nose_ok and ankles_ok:
            ankle_ys = [float(kps[i][0]) for i in (15, 16) if float(kps[i][2]) >= vis]
            if ankle_ys:
                dist = abs(float(np.mean(ankle_ys)) - float(kps[0][0]))
                if dist > 20:
                    self._samples_secondary.append(dist)

        if self._force_tertiary or (not ankles_ok and shoulders_ok and hips_ok):
            shoulder_ys = [float(kps[i][0]) for i in (5, 6) if float(kps[i][2]) >= vis]
            hip_ys      = [float(kps[i][0]) for i in (11, 12) if float(kps[i][2]) >= vis]
            if shoulder_ys and hip_ys:
                dist = abs(float(np.mean(hip_ys)) - float(np.mean(shoulder_ys)))
                if dist > 5:
                    self._samples_tertiary.append(dist)

        self._try_commit()

    @property
    def calib_progress(self) -> dict:
        """返回当前标定进度，供 UI 提示。"""
        if self._force_tertiary:
            return {"method": "shoulder_hip", "count": len(self._samples_tertiary), "target": MIN_CALIB_SAMPLES_TERTIARY}
        if self._samples_primary:
            return {"method": "shoulder_ankle", "count": len(self._samples_primary), "target": MIN_CALIB_SAMPLES}
        if self._samples_secondary:
            return {"method": "head_ankle", "count": len(self._samples_secondary), "target": MIN_CALIB_SAMPLES}
        if self._samples_tertiary:
            return {"method": "shoulder_hip", "count": len(self._samples_tertiary), "target": MIN_CALIB_SAMPLES_TERTIARY}
        return {"method": "waiting", "count": 0, "target": MIN_CALIB_SAMPLES}


# ──────────────────────────────────────────────────────────────
# 4. 深蹲动作状态机
# ──────────────────────────────────────────────────────────────

@dataclass
class RepResult:
    """一次完整深蹲的计算结果。"""
    rep_index: int              # 组内第几个 rep（从 1 开始）
    mcv: float                  # Mean Concentric Velocity (m/s)
    rom_m: float                # ROM（米）
    velocity_loss: Optional[float]  # 相对组内最佳 MCV 的速度损失率（%），首 rep 为 None
    concentric_frames: int      # 向心阶段帧数
    concentric_time_s: float    # 向心时间（秒）
    ascent_samples: list        # 逐帧瞬时速度列表
    peak_vel: float             # 向心阶段瞬时峰值速度
    start_frame: int            # 向心阶段起始帧索引
    end_frame: int              # 向心阶段结束帧索引
    y_bottom: float             # 最低点像素 Y
    y_top: float                # 结束时像素 Y
    rom_completion_pct: float   # ROM 完成百分比


class SquatStateMachine:
    """
    深蹲动作状态机。

    每帧调用 update(smoothed_y, frame_n, scale_m_per_px, video_fps)。
    当检测到完整 rep 时，finished_rep 属性会被设置为 RepResult，
    下一帧调用前需由调用方消费（并清空）。

    调用方负责：
      - 提供 smoothed_y（中值滤波后的髋部 Y 像素坐标）
      - 从 RepResult 读取结果后立即将 finished_rep 置为 None
    """

    def __init__(self, body_height_px: float, user_height_m: float, video_fps: float) -> None:
        self.body_height_px = float(body_height_px)
        self.user_height_m = float(user_height_m)
        self.video_fps = float(video_fps) if video_fps > 0 else 30.0

        self.state: str = "STANDING"
        self.running_avg_y: Optional[float] = None
        self.standing_baseline: Optional[float] = None
        self._baseline_samples: list = []
        self._baseline_locked = False
        self.MIN_BASELINE_SAMPLES = 10

        self.y_lowest: float = -1.0
        self.y_bottom: float = -1.0
        self.starting_height: float = -1.0
        self.start_asc_frame: int = -1
        self.frames_in_buffer: int = 0
        self.ascent_samples: list = []
        self.last_up_y: Optional[float] = None
        self.prev_inst_v: Optional[float] = None
        self.rom_completion_pct: float = 0.0

        self.rep_count: int = 0
        self.rep_velocities: list = []   # 每 rep 的 MCV
        self.best_mcv: float = 0.0

        self.finished_rep: Optional[RepResult] = None  # 消费后置 None

        self._movement_threshold = MOVEMENT_THRESHOLD_RATIO * self.body_height_px
        self._buffer_px = BUFFER_RATIO * self.body_height_px
        self._min_rom_m = MIN_ROM_RATIO * self.user_height_m

    # ── 公开接口 ──────────────────────────────────────────────

    def update(self, smoothed_y: float, frame_n: int, scale_m_per_px: float) -> None:
        """
        每帧调用。smoothed_y 为中值滤波后的髋部 Y（原图像素，向下增大）。
        scale_m_per_px: 标定完成后的比例尺。
        """
        self.finished_rep = None
        dt = 1.0 / self.video_fps

        # 维护站立基线
        if self.state == "STANDING":
            if not self._baseline_locked:
                self._baseline_samples.append(smoothed_y)
                if len(self._baseline_samples) >= self.MIN_BASELINE_SAMPLES:
                    self.standing_baseline = float(np.median(self._baseline_samples))
                    self._baseline_locked = True
            if self.running_avg_y is None:
                self.running_avg_y = smoothed_y
            else:
                self.running_avg_y = 0.97 * self.running_avg_y + 0.03 * smoothed_y

            if smoothed_y >= self.running_avg_y + self._movement_threshold:
                self.state = "DOWN"
                self.y_lowest = smoothed_y
                self.starting_height = (
                    self.standing_baseline if self.standing_baseline is not None
                    else self.running_avg_y
                )
                self.ascent_samples = []
                self.last_up_y = None
                self.prev_inst_v = None
                self.frames_in_buffer = 0
                logger.debug("frame %d: STANDING -> DOWN", frame_n)
            return

        if self.state == "DOWN":
            if smoothed_y > self.y_lowest:
                self.y_lowest = smoothed_y
            if smoothed_y < self.y_lowest - 2:
                self.state = "UP"
                self.y_bottom = self.y_lowest
                self.start_asc_frame = frame_n
                self.frames_in_buffer = 0
                self.last_up_y = smoothed_y
                self.prev_inst_v = None
                logger.debug("frame %d: DOWN -> UP, y_bottom=%.1f", frame_n, self.y_bottom)
            return

        if self.state == "UP":
            inst_v: Optional[float] = None
            if self.last_up_y is not None and scale_m_per_px > 0:
                raw_dy_px = abs(smoothed_y - self.last_up_y)
                if dt > 0:
                    inst_v = max(0.0, raw_dy_px * scale_m_per_px / dt)
                    self.ascent_samples.append(inst_v)
            self.last_up_y = smoothed_y

            # ROM 完成度
            denom = self.starting_height - self.y_lowest
            if abs(denom) > 1e-6:
                pct = (smoothed_y - self.y_lowest) / denom * 100.0
                self.rom_completion_pct = min(100.0, max(0.0, pct))

            # 早结束：速度开始下降且已完成 85%
            early_finish = False
            if inst_v is not None and self.prev_inst_v is not None and self.rom_completion_pct >= 85.0:
                if inst_v < self.prev_inst_v:
                    early_finish = True
            if inst_v is not None:
                self.prev_inst_v = inst_v

            ref_y = self.running_avg_y if self.running_avg_y is not None else self.starting_height
            in_buffer = abs(smoothed_y - ref_y) <= self._buffer_px
            self.frames_in_buffer = self.frames_in_buffer + 1 if in_buffer else 0
            rep_done = early_finish or (self.frames_in_buffer >= FRAMES_IN_BUFFER_TO_END)

            if rep_done:
                self._finish_rep(smoothed_y, frame_n, scale_m_per_px)

    # ── 内部 ──────────────────────────────────────────────────

    def _finish_rep(self, smoothed_y: float, frame_n: int, scale_m_per_px: float) -> None:
        concentric_frames = frame_n - self.start_asc_frame if self.start_asc_frame >= 0 else 0
        concentric_time = max(0.001, concentric_frames / self.video_fps)

        if concentric_time > 0.05 and self.y_bottom > 0:
            depth_m = abs(self.y_bottom - smoothed_y) * scale_m_per_px
            mcv = depth_m / concentric_time
            peak_v = float(max(self.ascent_samples)) if self.ascent_samples else 0.0

            if depth_m >= (self._min_rom_m - 0.005) and mcv >= (MIN_MCV_M_S - 0.005):
                self.rep_count += 1
                prev_best = self.best_mcv
                self.rep_velocities.append(mcv)
                self.best_mcv = max(self.rep_velocities)

                if self.rep_count == 1:
                    velocity_loss = None
                else:
                    velocity_loss = (
                        (self.best_mcv - mcv) / self.best_mcv * 100.0
                        if self.best_mcv > 0 else 0.0
                    )

                self.finished_rep = RepResult(
                    rep_index=self.rep_count,
                    mcv=mcv,
                    rom_m=depth_m,
                    velocity_loss=velocity_loss,
                    concentric_frames=concentric_frames,
                    concentric_time_s=concentric_time,
                    ascent_samples=list(self.ascent_samples),
                    peak_vel=peak_v,
                    start_frame=self.start_asc_frame,
                    end_frame=frame_n,
                    y_bottom=self.y_bottom,
                    y_top=smoothed_y,
                    rom_completion_pct=self.rom_completion_pct,
                )
                logger.info(
                    "Rep %d 完成: MCV=%.3f m/s ROM=%.2f m t=%.2fs",
                    self.rep_count, mcv, depth_m, concentric_time,
                )
            else:
                reasons = []
                if depth_m < self._min_rom_m:
                    reasons.append(f"ROM {depth_m*100:.0f}cm < {self._min_rom_m*100:.0f}cm")
                if mcv < MIN_MCV_M_S:
                    reasons.append(f"MCV {mcv:.3f} m/s < {MIN_MCV_M_S}")
                logger.debug("Rep 被拒绝: %s", "; ".join(reasons))

        # 重置
        self.state = "STANDING"
        self.y_lowest = -1.0
        self.y_bottom = -1.0
        self.starting_height = -1.0
        self.frames_in_buffer = 0
        self.running_avg_y = smoothed_y
        self.prev_inst_v = None
        self.rom_completion_pct = 0.0
        self.ascent_samples = []
        self.last_up_y = None
        self.start_asc_frame = -1


# ──────────────────────────────────────────────────────────────
# 5. 辅助：平滑跟踪点提取
# ──────────────────────────────────────────────────────────────

class HipTracker:
    """
    从关键点提取平滑后的髋部 Y/X 坐标。
    支持置信度过低时用历史帧插值。
    """
    CONF_THRESHOLD = 0.3

    def __init__(self, is_bodyweight: bool = False) -> None:
        self.is_bodyweight = is_bodyweight
        self._valid_y: deque = deque(maxlen=VALID_Y_HISTORY)
        self._med_hist: deque = deque(maxlen=MEDIAN_WINDOW)

    def update(self, kps: np.ndarray) -> tuple:
        """
        Args:
            kps: shape (17, 3), 原图像素坐标
        Returns:
            (smoothed_y, raw_y, raw_x)  — 其中任意值可能为 None
        """
        th = self.CONF_THRESHOLD
        conf9  = float(kps[9][2])  if len(kps) > 9  else 0.0
        conf10 = float(kps[10][2]) if len(kps) > 10 else 0.0
        conf11 = float(kps[11][2]) if len(kps) > 11 else 0.0
        conf12 = float(kps[12][2]) if len(kps) > 12 else 0.0

        raw_y: Optional[float] = None
        raw_x: Optional[float] = None

        if self.is_bodyweight:
            if conf11 >= th and conf12 >= th:
                raw_y = (float(kps[11][0]) + float(kps[12][0])) / 2.0
                raw_x = (float(kps[11][1]) + float(kps[12][1])) / 2.0
        else:
            if conf9 >= th and conf10 >= th:
                raw_y = (float(kps[9][0]) + float(kps[10][0])) / 2.0
                raw_x = (float(kps[9][1]) + float(kps[10][1])) / 2.0
            elif conf11 >= th:
                raw_y = float(kps[11][0])
                raw_x = float(kps[11][1])
            elif conf12 >= th:
                raw_y = float(kps[12][0])
                raw_x = float(kps[12][1])
            elif self._valid_y:
                raw_y = float(np.mean(self._valid_y))
                raw_x = None

        if raw_y is not None:
            self._valid_y.append(raw_y)
            self._med_hist.append(raw_y)
            smoothed = float(np.median(self._med_hist)) if len(self._med_hist) >= MEDIAN_WINDOW else raw_y
        else:
            smoothed = None

        return smoothed, raw_y, raw_x

    def clear(self) -> None:
        self._valid_y.clear()
        self._med_hist.clear() 