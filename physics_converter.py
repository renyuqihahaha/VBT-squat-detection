#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhysicsConverter - 物理转换逻辑封装
- 像素 ↔ 真实物理尺寸（米/厘米）
- 身高标定、Plane-to-Plane 深度偏移量
仿照 helperFunctions 的模块化结构，集中管理所有物理换算。
"""


class PhysicsConverter:
    """物理转换：像素转米/厘米、深度偏移量（Plane-to-Plane）。"""

    # 深度过滤阈值：±0.5 cm 内视为无效波动（MoveNet 抖动）
    DEPTH_JITTER_THRESHOLD_CM = 0.5
    # 体型补偿系数：final_depth = calculated + BIAS_CM
    BIAS_CM = 2.0
    # 未标定时的 fallback 系数（1080p）
    DEFAULT_SCALE = 0.0031
    # 无用户身高时的兜底值（米），仅用于兼容旧调用
    _FALLBACK_HEIGHT_M = 1.75

    def __init__(self, real_height_m=None):
        """
        Args:
            real_height_m: 用户真实身高（米），必须由调用方传入。不传时从 vbt_runtime_config 读取。
        """
        if real_height_m is not None:
            self._real_height_m = float(real_height_m)
        else:
            try:
                from vbt_runtime_config import get_user_height_cm
                cm = get_user_height_cm()
                self._real_height_m = (cm / 100.0) if cm and 100 <= cm <= 250 else self._FALLBACK_HEIGHT_M
            except Exception:
                self._real_height_m = self._FALLBACK_HEIGHT_M
        self._m_per_pixel = None  # 标定后缓存

    def calculate_m_per_pixel(self, pixel_height):
        """
        根据站立时头顶到脚踝像素距离计算 m_per_pixel。
        m_per_pixel = real_height_m / pixel_height（real_height_m 来自构造参数或 vbt_runtime_config）
        Args:
            pixel_height: 头顶到脚踝的垂直像素距离（px）
        Returns:
            像素位移 × 该值 = 真实位移（m），无效则 None
        """
        if pixel_height is None or pixel_height <= 0:
            return None
        return self._real_height_m / pixel_height

    def pixel_to_m(self, pixel_value):
        """像素值转米。需先通过 calculate_m_per_pixel 获得 m_per_pixel。"""
        if self._m_per_pixel is None or pixel_value is None:
            return None
        return pixel_value * self._m_per_pixel

    def pixel_to_cm(self, pixel_value):
        """像素值转厘米。"""
        m = self.pixel_to_m(pixel_value)
        return m * 100 if m is not None else None

    def depth_offset_cm(self, mid_hip_y_px, mid_knee_y_px, m_per_pixel=None):
        """
        Plane-to-Plane 深度偏移量（厘米）。
        公式：depth_offset = (Y_mid_hip - Y_mid_knee) × m_per_pixel × 100
        坐标系：OpenCV Y 向下，髋低于膝 → Y_hip > Y_knee → 正值表示达标。
        Args:
            mid_hip_y_px: 左右髋部中点 Y（像素）
            mid_knee_y_px: 左右膝盖中点 Y（像素）
            m_per_pixel: 像素-米转换系数，若不传则用实例缓存的 _m_per_pixel
        Returns:
            偏移量（厘米），正=髋低于膝（达标），负=未达标；±0.5 cm 内舍为 0
        """
        mp = m_per_pixel if m_per_pixel is not None else self._m_per_pixel
        if mid_hip_y_px is None or mid_knee_y_px is None or mp is None or mp <= 0:
            return None
        depth_offset_px = mid_hip_y_px - mid_knee_y_px
        depth_offset_cm = depth_offset_px * mp * 100
        if abs(depth_offset_cm) < PhysicsConverter.DEPTH_JITTER_THRESHOLD_CM:
            depth_offset_cm = 0.0
        return depth_offset_cm + PhysicsConverter.BIAS_CM

    def set_m_per_pixel(self, value):
        """设置并缓存 m_per_pixel。"""
        self._m_per_pixel = value


# ================= 便捷函数（兼容现有调用） =================
# 使用默认实例（从 vbt_runtime_config 读取 user_height_cm），保持导入方式不变
_default_converter = PhysicsConverter()

DEFAULT_SCALE = PhysicsConverter.DEFAULT_SCALE
DEPTH_JITTER_THRESHOLD_CM = PhysicsConverter.DEPTH_JITTER_THRESHOLD_CM


def calculate_m_per_pixel(pixel_height):
    """兼容：根据 pixel_height 计算 m_per_pixel。"""
    return _default_converter.calculate_m_per_pixel(pixel_height)


def get_depth_offset(mid_hip_y_px, mid_knee_y_px, m_per_pixel):
    """兼容：Plane-to-Plane 深度偏移量。参数名已更新为 midpoint。"""
    return _default_converter.depth_offset_cm(mid_hip_y_px, mid_knee_y_px, m_per_pixel)


def pixel_displacement_to_velocity_m_per_s(
    delta_px: float, scale_m_per_px: float, dt_s: float
) -> float:
    """
    像素位移转真实速度 (m/s)。
    公式：V = dy_px × Ratio / dt  (dy_px 必须为 _unpad_keypoint 还原后的像素值)
    Args:
        delta_px: 垂直位移（像素，Y 轴向下为正，须为还原后真实像素）
        scale_m_per_px: 像素-米转换系数 (m/px)，168cm 用户合理范围 0.003-0.006
        dt_s: 时间间隔（秒）
    Returns:
        速度 (m/s)，dt_s <= 0 时返回 0.0
    """
    if dt_s is None or dt_s <= 0 or scale_m_per_px is None or scale_m_per_px <= 0:
        return 0.0
    v = max(0.0, float(delta_px) * float(scale_m_per_px) / float(dt_s))
    return v
