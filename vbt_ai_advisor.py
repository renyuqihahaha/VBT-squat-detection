#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBT AI 智能教练：LVP 负荷-速度建模、1RM 预测、个性化训练建议引擎。
使用 numpy 线性回归拟合，避免对 sklearn 的强依赖以适配树莓派。
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from vbt_analytics_pro import DB_PATH

logger = logging.getLogger("vbt_ai_advisor")

MVT_M_S = 0.30  # 深蹲最小速度阈值 (Minimum Velocity Threshold)
MIN_DATA_POINTS = 2  # 建模所需最少不同重量数据点（3个更佳，2个即可初步拟合）
ROM_QUALITY_THRESHOLD = 80.0  # ROM% 过滤阈值


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class LVPModel:
    """负荷-速度线性模型: V = slope * Load + intercept"""
    slope: float
    intercept: float
    r_squared: float
    predicted_1rm_kg: float
    data_points: int
    loads: list[float] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)


@dataclass
class DailyReadiness:
    """当日状态评估"""
    score: float  # 0-100
    velocity_deviation_pct: float  # 正=超常，负=疲劳
    status: str  # "optimal" | "normal" | "fatigued"
    message: str


@dataclass
class TrainingAdvice:
    """单条训练建议"""
    action: str  # "continue" | "add_weight" | "reduce_weight" | "rest" | "stop"
    message: str
    recommended_load_kg: Optional[float]
    velocity_zone: str  # "too_fast" | "optimal" | "too_slow" | "fatigued"


# ---------------------------------------------------------------------------
# LVP 建模（纯 numpy，无需 sklearn）
# ---------------------------------------------------------------------------

MIN_VALID_VELOCITY = 0.10  # 低于此阈值的平均速度视为标定异常


def _fetch_lvp_data(db_path: str, user_name: str = "qiqi") -> tuple[list[float], list[float]]:
    """
    从 DB 读取有效数据用于 LVP 拟合。
    强制使用平均速度 (v_mean)。
    过滤: load_kg > 0, v_mean >= MIN_VALID_VELOCITY, rom_completion_pct >= 80 (或 NULL 时不过滤)。
    按 load_kg 分组取均值。
    """
    uname = (user_name or "qiqi").strip() or "qiqi"
    uname = "_".join(uname.split())
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT load_kg, AVG(v_mean) AS avg_v, COUNT(*) AS cnt
        FROM reps
        WHERE user_name = ?
          AND load_kg IS NOT NULL AND load_kg > 0
          AND v_mean IS NOT NULL AND v_mean >= ?
          AND (rom_completion_pct IS NULL OR rom_completion_pct >= ?)
        GROUP BY load_kg
        HAVING cnt >= 1
        ORDER BY load_kg ASC
        """,
        (uname, MIN_VALID_VELOCITY, ROM_QUALITY_THRESHOLD),
    )
    rows = cur.fetchall()
    conn.close()

    loads = [float(r[0]) for r in rows]
    velocities = [float(r[1]) for r in rows]
    return loads, velocities


def _fetch_lvp_data_peak(db_path: str, user_name: str = "qiqi") -> tuple[list[float], list[float]]:
    """
    从 DB 读取有效数据，按 load_kg 分组取 MAX(v_mean) 巅峰速度。
    与 1RM 面板、批量校准逻辑一致，确保全 App 数据源统一。
    """
    uname = (user_name or "qiqi").strip() or "qiqi"
    uname = "_".join(uname.split())
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT load_kg, MAX(v_mean) AS peak_v
        FROM reps
        WHERE user_name = ?
          AND load_kg IS NOT NULL AND load_kg > 0
          AND v_mean IS NOT NULL AND v_mean >= ?
          AND (rom_completion_pct IS NULL OR rom_completion_pct >= ?)
        GROUP BY load_kg
        ORDER BY load_kg ASC
        """,
        (uname, MIN_VALID_VELOCITY, ROM_QUALITY_THRESHOLD),
    )
    rows = cur.fetchall()
    conn.close()

    loads = [float(r[0]) for r in rows]
    velocities = [float(r[1]) for r in rows]
    return loads, velocities


def get_unified_lvp_prediction(
    db_path: str = DB_PATH,
    user_name: str = "qiqi",
    target_date_str: Optional[str] = None,
) -> Optional[LVP1RMPrediction]:
    """
    统一 1RM 预测入口（Single Source of Truth）。
    使用巅峰速度提取、_clean_lvp_data 单调性去噪、predict_1rm_time_weighted。
    供 AI 智能教练、1RM 面板等所有展示 1RM 的模块调用。
    target_date_str: 若指定则按单日拟合；否则按汇总模式（全部历史 max 聚合）。
    """
    loads, velocities = _fetch_lvp_data_peak(db_path, user_name)
    if len(loads) < MIN_DATA_POINTS:
        return None

    date_str = target_date_str or datetime.now().strftime("%Y-%m-%d")
    return predict_1rm_time_weighted(
        loads, velocities, date_str,
        history_rows=None,
    )


def build_lvp_model(db_path: str = DB_PATH, user_name: str = "qiqi") -> Optional[LVPModel]:
    """
    构建 LVP 线性回归模型（旧版，使用 AVG 聚合，无去噪）。
    @deprecated 请使用 get_unified_lvp_prediction() 以与 1RM 面板保持一致。

    边界保护:
    - Rule 1: slope 必须 < 0 (负重越大速度越慢)，否则返回 None
    - Rule 2: 所有速度均 < MVT 时，说明标定异常，返回 None
    - Rule 3: R² < 0.65 时标记警告但仍返回模型（由 UI 决定是否显示）
    - 预测 1RM 必须 > 0 且 < 1000kg，否则视为无效
    """
    loads, velocities = _fetch_lvp_data(db_path, user_name)
    if len(loads) < MIN_DATA_POINTS:
        logger.info("LVP 数据不足: 仅 %d 个不同重量点 (需要 %d)", len(loads), MIN_DATA_POINTS)
        return None

    # Rule 2: 所有速度均低于 MVT → 标定异常
    if all(v < MVT_M_S for v in velocities):
        logger.warning("LVP 拦截: 所有平均速度 (%.3f ~ %.3f) 均低于 MVT (%.2f)，疑似标定异常",
                        min(velocities), max(velocities), MVT_M_S)
        return None

    x = np.asarray(loads, dtype=np.float64)
    y = np.asarray(velocities, dtype=np.float64)

    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if abs(ss_xx) < 1e-12:
        logger.warning("LVP 拟合失败: 负荷数据方差为零")
        return None

    slope = float(ss_xy / ss_xx)
    intercept = float(y_mean - slope * x_mean)

    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if abs(ss_tot) > 1e-12 else 0.0

    # Rule 1: slope 必须为负 (越重越慢)
    if slope >= 0:
        logger.warning("LVP 拦截: slope=%.6f >= 0 (越重越快)，数据异常或负荷范围不足", slope)
        return None

    predicted_1rm = (MVT_M_S - intercept) / slope

    # 物理常识: 1RM 必须为正且不超过 1000kg
    if predicted_1rm <= 0 or predicted_1rm > 1000:
        logger.warning("LVP 拦截: 预测 1RM=%.1f kg 超出合理范围 (0, 1000]", predicted_1rm)
        return None

    if r_squared < 0.65:
        logger.info("LVP 警告: R²=%.3f < 0.65，数据线性度较差，预测仅供参考", r_squared)

    return LVPModel(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        predicted_1rm_kg=predicted_1rm,
        data_points=n,
        loads=loads,
        velocities=velocities,
    )


# ---------------------------------------------------------------------------
# 时间加权自学习 1RM 预测 (Time-Weighted Historical Smoothing)
# ---------------------------------------------------------------------------

R2_THRESHOLD_AI_FALLBACK = 0.75  # 单日 R² 低于此值时触发历史融合
TIME_DECAY_HALF_LIFE_DAYS = 10.0  # 时间衰减半衰期（天）
MONOTONIC_REL_THRESHOLD = 0.05  # 相对容差：较重负荷速度高于较轻 > 5% 则剔除
MONOTONIC_ABS_THRESHOLD = 0.02  # 绝对容差：高出 > 0.02 m/s 则剔除


@dataclass
class LVP1RMPrediction:
    """1RM 预测结果（含融合与保底标记）"""
    slope: float
    intercept: float
    r_squared: float
    predicted_1rm_kg: float
    mvt_baseline: float
    loads: list[float]
    velocities: list[float]
    historical_fusion_used: bool = False
    uses_polynomial_fallback: bool = False
    outliers_dropped: bool = False


def _clean_lvp_data(
    loads: list[float],
    velocities: list[float],
    rel_threshold: float = MONOTONIC_REL_THRESHOLD,
    abs_threshold: float = MONOTONIC_ABS_THRESHOLD,
) -> tuple[list[float], list[float], bool]:
    """
    单调性抗噪过滤：LVP 应严格单调递减（负重↑速度↓）。
    若较重负荷速度异常高于较轻负荷，且高出幅度 > 5% 或 > 0.02 m/s，则剔除该异常点。
    返回 (cleaned_loads, cleaned_velocities, outliers_dropped)。
    """
    if len(loads) < 2:
        return loads, velocities, False
    pairs = sorted(zip(loads, velocities), key=lambda p: p[0])
    out_loads: list[float] = [pairs[0][0]]
    out_vels: list[float] = [pairs[0][1]]
    dropped = False
    for i in range(1, len(pairs)):
        load_i, vel_i = pairs[i]
        load_prev, vel_prev = out_loads[-1], out_vels[-1]
        if load_i <= load_prev:
            continue
        if vel_i > vel_prev:
            rel_inc = (vel_i - vel_prev) / vel_prev if vel_prev > 1e-9 else 0.0
            abs_inc = vel_i - vel_prev
            if rel_inc > rel_threshold or abs_inc > abs_threshold:
                dropped = True
                continue
        out_loads.append(load_i)
        out_vels.append(vel_i)
    return out_loads, out_vels, dropped


def _filter_monotonic_with_weights(
    loads: list[float],
    velocities: list[float],
    weights: list[float],
    rel_threshold: float = MONOTONIC_REL_THRESHOLD,
    abs_threshold: float = MONOTONIC_ABS_THRESHOLD,
) -> tuple[list[float], list[float], list[float]]:
    """单调性过滤，同步保留 weights。"""
    if len(loads) < 2:
        return loads, velocities, weights
    triples = sorted(zip(loads, velocities, weights), key=lambda t: t[0])
    out_loads, out_vels, out_weights = [triples[0][0]], [triples[0][1]], [triples[0][2]]
    for i in range(1, len(triples)):
        load_i, vel_i, w_i = triples[i]
        load_prev, vel_prev = out_loads[-1], out_vels[-1]
        if load_i <= load_prev:
            continue
        if vel_i > vel_prev:
            rel_inc = (vel_i - vel_prev) / vel_prev if vel_prev > 1e-9 else 0.0
            abs_inc = vel_i - vel_prev
            if rel_inc > rel_threshold or abs_inc > abs_threshold:
                continue
        out_loads.append(load_i)
        out_vels.append(vel_i)
        out_weights.append(w_i)
    return out_loads, out_vels, out_weights


def predict_1rm_time_weighted(
    loads_today: list[float],
    velocities_today: list[float],
    target_date_str: str,
    history_rows: Optional[list[tuple[float, float, str]]] = None,
    min_r2_fallback: float = R2_THRESHOLD_AI_FALLBACK,
    decay_half_life: float = TIME_DECAY_HALF_LIFE_DAYS,
) -> Optional[LVP1RMPrediction]:
    """
    时间加权自学习 1RM 预测。
    - 单调性过滤：剔除违反负荷-速度单调递减的离群点。
    - 先对单日数据做线性拟合，计算 R²。
    - 若 R² < min_r2_fallback 且存在历史数据，则合并并做时间衰减加权回归 (WLS)。
    - 若 slope >= 0，尝试二次多项式拟合，取其切线或极限作为保底。
    history_rows: [(load_kg, v_mean, date_str), ...]，过去 30 天内的有效数据。
    """
    loads_today, velocities_today, outliers_dropped = _clean_lvp_data(loads_today, velocities_today)
    if len(loads_today) < MIN_DATA_POINTS or len(velocities_today) < MIN_DATA_POINTS:
        return None

    x_today = np.asarray(loads_today, dtype=np.float64)
    y_today = np.asarray(velocities_today, dtype=np.float64)

    # MVT 硬编码：深蹲标准阈值为 0.30 m/s，红色底线固定
    mvt_baseline = float(MVT_M_S)

    # Step 1: 单日拟合
    slope, intercept, r_sq, historical_fusion, poly_fallback = _fit_daily_then_fallback(
        x_today, y_today, target_date_str, history_rows, min_r2_fallback, decay_half_life,
    )
    if slope is None:
        return None

    # 1RM 计算
    if abs(slope) < 1e-12:
        return None
    predicted_1rm = (mvt_baseline - intercept) / slope
    if predicted_1rm <= 0 or predicted_1rm > 1000:
        return None

    return LVP1RMPrediction(
        slope=slope,
        intercept=intercept,
        r_squared=r_sq,
        predicted_1rm_kg=predicted_1rm,
        mvt_baseline=mvt_baseline,
        loads=loads_today,
        velocities=velocities_today,
        historical_fusion_used=historical_fusion,
        uses_polynomial_fallback=poly_fallback,
        outliers_dropped=outliers_dropped,
    )


def _fit_daily_then_fallback(
    x_today: np.ndarray,
    y_today: np.ndarray,
    target_date_str: str,
    history_rows: Optional[list[tuple[float, float, str]]],
    min_r2: float,
    decay_hl: float,
) -> tuple[Optional[float], Optional[float], float, bool, bool]:
    """
    返回 (slope, intercept, r_squared, historical_fusion_used, polynomial_fallback_used)。
    """
    # 单日 OLS
    coeffs = np.polyfit(x_today, y_today, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    y_pred = slope * x_today + intercept
    ss_res = float(np.sum((y_today - y_pred) ** 2))
    ss_tot = float(np.sum((y_today - np.mean(y_today)) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if abs(ss_tot) > 1e-12 else 0.0

    historical_fusion = False
    poly_fallback = False

    # AI Fallback: R² < 0.75 且存在历史数据
    if r_sq < min_r2 and history_rows and len(history_rows) >= 1:
        historical_fusion = True
        try:
            target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        except ValueError:
            target_dt = datetime.now()

        # 合并数据并计算权重
        all_loads: list[float] = list(x_today)
        all_vels: list[float] = list(y_today)
        all_weights: list[float] = [1.0] * len(x_today)  # 当日权重为 1

        for load_kg, v_mean, date_str in history_rows:
            try:
                row_dt = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            days_diff = (target_dt - row_dt).days
            if days_diff < 0:
                continue
            weight = float(np.exp(-days_diff / decay_hl))
            all_loads.append(float(load_kg))
            all_vels.append(float(v_mean))
            all_weights.append(weight)

        all_loads, all_vels, all_weights = _filter_monotonic_with_weights(
            all_loads, all_vels, all_weights,
        )

        x_all = np.asarray(all_loads, dtype=np.float64)
        y_all = np.asarray(all_vels, dtype=np.float64)
        w_all = np.asarray(all_weights, dtype=np.float64)

        if len(x_all) < MIN_DATA_POINTS:
            pass  # 保持单日结果
        else:
            coeffs = np.polyfit(x_all, y_all, 1, w=w_all)
            slope = float(coeffs[0])
            intercept = float(coeffs[1])
            y_pred_all = slope * x_all + intercept
            ss_res = float(np.sum(w_all * (y_all - y_pred_all) ** 2))
            ss_tot = float(np.sum(w_all * (y_all - np.average(y_all, weights=w_all)) ** 2))
            r_sq = 1.0 - ss_res / ss_tot if abs(ss_tot) > 1e-12 else r_sq

    # 物理常识过滤: slope >= 0 时尝试二次多项式保底
    if slope >= 0:
        poly_fallback = True
        slope, intercept = _polynomial_fallback_fit(x_today, y_today, history_rows, target_date_str, decay_hl)
        if slope is None:
            return None, None, r_sq, historical_fusion, True

    return slope, intercept, r_sq, historical_fusion, poly_fallback


def _polynomial_fallback_fit(
    x_today: np.ndarray,
    y_today: np.ndarray,
    history_rows: Optional[list[tuple[float, float, str]]],
    target_date_str: str,
    decay_hl: float,
) -> tuple[Optional[float], Optional[float]]:
    """
    当线性斜率 >= 0 时，尝试二次多项式拟合，取 V=MVT 处的切线斜率作为等效 slope，
    或取抛物线在合理负重区间的极限。返回 (slope, intercept) 或 (None, None)。
    """
    # 优先用合并后的数据（若有历史）
    x = np.asarray(list(x_today), dtype=np.float64)
    y = np.asarray(list(y_today), dtype=np.float64)
    w = np.ones_like(x)

    if history_rows and len(history_rows) >= 1:
        try:
            target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
        except ValueError:
            target_dt = datetime.now()
        for load_kg, v_mean, date_str in history_rows:
            try:
                row_dt = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            days_diff = (target_dt - row_dt).days
            if days_diff < 0:
                continue
            weight = float(np.exp(-days_diff / decay_hl))
            x = np.append(x, float(load_kg))
            y = np.append(y, float(v_mean))
            w = np.append(w, weight)

    if len(x) < 3:
        return None, None

    # 二次拟合: V = a*L^2 + b*L + c，期望 a < 0（开口向下）
    coeffs = np.polyfit(x, y, 2, w=w)
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    # 若 a >= 0，抛物线开口向上，物理上不合理，用负斜率近似
    if a >= 0:
        # 退化为过最重负荷点的负斜率直线
        idx_max_load = int(np.argmax(x))
        slope_approx = -0.001  # 保底负斜率
        intercept_approx = float(y[idx_max_load]) - slope_approx * float(x[idx_max_load])
        return slope_approx, intercept_approx

    # 在最大负荷处取切线: dV/dL = 2a*L + b, V(L_max) = a*L_max^2 + b*L_max + c
    load_max = float(np.max(x))
    slope_tangent = 2 * a * load_max + b
    if slope_tangent >= 0:
        slope_tangent = -0.001
    v_at_max = a * load_max * load_max + b * load_max + c
    intercept_tangent = v_at_max - slope_tangent * load_max
    return slope_tangent, intercept_tangent


# ---------------------------------------------------------------------------
# 当日状态评估
# ---------------------------------------------------------------------------

ABSOLUTE_MVT_DANGER = 0.35  # 低于此速度视为力竭边缘，强制报警


def assess_daily_readiness(
    db_path: str = DB_PATH,
    user_name: str = "qiqi",
    current_load_kg: float = 0.0,
    current_first_set_velocity: float = 0.0,
) -> DailyReadiness:
    """
    计算当日状态评分：将当前首组速度与历史同重量下的速度对比。
    偏差 > +5%: 超常 (optimal)
    偏差 -5% ~ +5%: 正常 (normal)
    偏差 < -10%: 疲劳 (fatigued)
    绝对阈值: 速度 <= 0.35 m/s 时强制返回 fatigued（1RM 极限试举等场景）。
    """
    if current_load_kg <= 0 or current_first_set_velocity <= 0:
        return DailyReadiness(
            score=50.0, velocity_deviation_pct=0.0,
            status="normal", message="数据不足，无法评估当日状态",
        )

    # 绝对阈值拦截：单次 1RM 极限试举等场景，无历史对比时流失率为 0%，但速度极低
    if current_first_set_velocity <= ABSOLUTE_MVT_DANGER:
        return DailyReadiness(
            score=10.0,
            velocity_deviation_pct=-100.0,
            status="fatigued",
            message="⚠ 绝对速度极低，已达力竭边缘 (RPE 9.5-10)，强烈建议停止！",
        )

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    today_str = datetime.now().strftime("%Y-%m-%d")
    uname = (user_name or "qiqi").strip() or "qiqi"
    uname = "_".join(uname.split())
    cur.execute(
        """
        SELECT AVG(v_mean) FROM reps
        WHERE user_name = ?
          AND load_kg = ?
          AND v_mean > 0.02
          AND substr(ts, 1, 10) != ?
        """,
        (uname, float(current_load_kg), today_str),
    )
    row = cur.fetchone()
    conn.close()

    hist_avg = float(row[0]) if row and row[0] is not None else 0.0
    if hist_avg <= 0:
        return DailyReadiness(
            score=50.0, velocity_deviation_pct=0.0,
            status="normal", message="历史参考数据不足",
        )

    deviation = (current_first_set_velocity - hist_avg) / hist_avg * 100.0

    if deviation > 5.0:
        score = min(100.0, 70.0 + deviation)
        status = "optimal"
        message = f"状态极佳，速度高于历史均值 {deviation:.1f}%"
    elif deviation > -10.0:
        score = 50.0 + deviation * 2
        status = "normal"
        message = f"状态正常，速度偏差 {deviation:+.1f}%"
    else:
        score = max(0.0, 50.0 + deviation * 2)
        status = "fatigued"
        message = f"中枢神经疲劳，速度低于历史均值 {abs(deviation):.1f}%，建议降低强度"

    return DailyReadiness(
        score=round(score, 1),
        velocity_deviation_pct=round(deviation, 1),
        status=status,
        message=message,
    )


# ---------------------------------------------------------------------------
# 训练建议引擎
# ---------------------------------------------------------------------------

STRENGTH_VELOCITY_RANGE = (0.15, 0.50)
STRENGTH_LOSS_WARN = 10.0
STRENGTH_LOSS_STOP = 20.0

HYPERTROPHY_VELOCITY_RANGE = (0.50, 0.75)
HYPERTROPHY_LOSS_EFFECTIVE = 30.0
HYPERTROPHY_LOSS_STOP = 50.0


def get_training_advice(
    current_velocity: float,
    velocity_loss_pct: float,
    mode: str,
    lvp_model: Optional[LVPModel] = None,
    current_load_kg: float = 0.0,
) -> TrainingAdvice:
    """
    根据训练模式生成实时建议。
    mode: "strength" (增力) 或 "hypertrophy" (肌肥大)
    绝对阈值: 速度 <= 0.35 m/s 时强制报警，无论流失率。
    """
    recommended_load: Optional[float] = None

    # 绝对阈值拦截：1RM 极限试举等场景，流失率可能为 0% 但速度极低
    if current_velocity > 0 and current_velocity <= ABSOLUTE_MVT_DANGER:
        return TrainingAdvice(
            action="stop",
            message="⚠ 绝对速度极低，已达力竭边缘 (RPE 9.5-10)，强烈建议停止！",
            recommended_load_kg=None,
            velocity_zone="fatigued",
        )

    if mode == "strength":
        v_low, v_high = STRENGTH_VELOCITY_RANGE
        if velocity_loss_pct >= STRENGTH_LOSS_STOP:
            return TrainingAdvice(
                action="stop",
                message=f"速度流失 {velocity_loss_pct:.0f}% 超过增力阈值 {STRENGTH_LOSS_STOP:.0f}%，强烈建议停止本组",
                recommended_load_kg=None,
                velocity_zone="fatigued",
            )
        if velocity_loss_pct >= STRENGTH_LOSS_WARN:
            return TrainingAdvice(
                action="rest",
                message=f"速度流失 {velocity_loss_pct:.0f}%，接近增力极限，建议增加组间休息至 3-5 分钟",
                recommended_load_kg=None,
                velocity_zone="fatigued",
            )
        if current_velocity > v_high:
            if lvp_model and abs(lvp_model.slope) > 1e-9:
                target_v = (v_low + v_high) / 2
                recommended_load = (target_v - lvp_model.intercept) / lvp_model.slope
                recommended_load = max(0.0, round(recommended_load / 2.5) * 2.5)
            return TrainingAdvice(
                action="add_weight",
                message=f"当前速度 {current_velocity:.2f} m/s 高于增力区间上限，建议加重",
                recommended_load_kg=recommended_load,
                velocity_zone="too_fast",
            )
        if current_velocity < v_low:
            return TrainingAdvice(
                action="reduce_weight",
                message=f"当前速度 {current_velocity:.2f} m/s 低于增力区间下限，建议减重以保证爆发力",
                recommended_load_kg=None,
                velocity_zone="too_slow",
            )
        return TrainingAdvice(
            action="continue",
            message=f"速度 {current_velocity:.2f} m/s 处于增力最佳区间 ({v_low}-{v_high} m/s)，继续保持",
            recommended_load_kg=current_load_kg if current_load_kg > 0 else None,
            velocity_zone="optimal",
        )

    else:  # hypertrophy
        v_low, v_high = HYPERTROPHY_VELOCITY_RANGE
        if velocity_loss_pct >= HYPERTROPHY_LOSS_STOP:
            return TrainingAdvice(
                action="stop",
                message=f"速度流失 {velocity_loss_pct:.0f}% 达到肌肥大极限 {HYPERTROPHY_LOSS_STOP:.0f}%，本组已完成",
                recommended_load_kg=None,
                velocity_zone="fatigued",
            )
        if velocity_loss_pct >= HYPERTROPHY_LOSS_EFFECTIVE:
            return TrainingAdvice(
                action="continue",
                message=f"速度流失 {velocity_loss_pct:.0f}% 进入有效代谢压力区间，这是肌肥大的黄金区域",
                recommended_load_kg=current_load_kg if current_load_kg > 0 else None,
                velocity_zone="optimal",
            )
        if current_velocity > v_high:
            if lvp_model and abs(lvp_model.slope) > 1e-9:
                target_v = (v_low + v_high) / 2
                recommended_load = (target_v - lvp_model.intercept) / lvp_model.slope
                recommended_load = max(0.0, round(recommended_load / 2.5) * 2.5)
            return TrainingAdvice(
                action="add_weight",
                message=f"当前速度 {current_velocity:.2f} m/s 过快，未达肌肥大刺激区间，建议加重",
                recommended_load_kg=recommended_load,
                velocity_zone="too_fast",
            )
        if current_velocity < v_low:
            return TrainingAdvice(
                action="reduce_weight",
                message=f"当前速度 {current_velocity:.2f} m/s 过低，可能导致技术变形，建议适当减重",
                recommended_load_kg=None,
                velocity_zone="too_slow",
            )
        return TrainingAdvice(
            action="continue",
            message=f"速度 {current_velocity:.2f} m/s 处于肌肥大最佳区间 ({v_low}-{v_high} m/s)，继续积累代谢压力",
            recommended_load_kg=current_load_kg if current_load_kg > 0 else None,
            velocity_zone="optimal",
        )


def get_recommended_load(
    lvp_model: Optional[LVPModel],
    mode: str,
) -> Optional[float]:
    """根据 LVP 模型与训练模式，推算推荐负荷 (kg)。"""
    if lvp_model is None or abs(lvp_model.slope) < 1e-9:
        return None

    if mode == "strength":
        target_v = (STRENGTH_VELOCITY_RANGE[0] + STRENGTH_VELOCITY_RANGE[1]) / 2
    else:
        target_v = (HYPERTROPHY_VELOCITY_RANGE[0] + HYPERTROPHY_VELOCITY_RANGE[1]) / 2

    load = (target_v - lvp_model.intercept) / lvp_model.slope
    if load <= 0:
        return None
    return round(load / 2.5) * 2.5


# ---------------------------------------------------------------------------
# Pose-Guard 姿态诊断
# ---------------------------------------------------------------------------

KNEE_VALGUS_THRESHOLD = 0.85
TORSO_LEAN_WARNING_DEG = 45.0
STABILITY_DRIFT_RATIO = 0.10


@dataclass
class PoseDiagnosis:
    """单帧姿态诊断结果"""
    knee_valgus: bool = False
    knee_ratio: float = 1.0
    torso_lean_warning: bool = False
    torso_angle: float = 0.0
    stability_warning: bool = False
    hip_x_drift_ratio: float = 0.0
    issues: list[str] = field(default_factory=list)
    score: float = 100.0


def diagnose_pose(
    kps: np.ndarray,
    w: int,
    hip_x_history: list[float],
    body_width_px: float,
    phase: str,
) -> PoseDiagnosis:
    """
    实时姿态诊断（纯 numpy 向量化）。
    kps: 必须为 _unpad_keypoint 还原后的原视频分辨率绝对像素坐标，每行 [y_px, x_px, confidence]。
    """
    diag = PoseDiagnosis()
    issues: list[str] = []
    penalty = 0.0

    lk_conf = float(kps[13][2])
    rk_conf = float(kps[14][2])
    la_conf = float(kps[15][2])
    ra_conf = float(kps[16][2])

    if lk_conf > 0.3 and rk_conf > 0.3 and la_conf > 0.3 and ra_conf > 0.3:
        knee_dx = abs(float(kps[13][1]) - float(kps[14][1]))
        ankle_dx = abs(float(kps[15][1]) - float(kps[16][1]))
        if ankle_dx > 5:
            ratio = knee_dx / ankle_dx
            diag.knee_ratio = round(ratio, 3)
            if ratio < KNEE_VALGUS_THRESHOLD and phase in ("DOWN", "UP"):
                diag.knee_valgus = True
                issues.append("knee_valgus")
                penalty += 20.0

    ls_conf = float(kps[5][2])
    rs_conf = float(kps[6][2])
    lh_conf = float(kps[11][2])
    rh_conf = float(kps[12][2])
    if ls_conf > 0.3 and rs_conf > 0.3 and lh_conf > 0.3 and rh_conf > 0.3:
        shoulder_x = (float(kps[5][1]) + float(kps[6][1])) / 2.0
        shoulder_y = (float(kps[5][0]) + float(kps[6][0])) / 2.0
        hip_x = (float(kps[11][1]) + float(kps[12][1])) / 2.0
        hip_y = (float(kps[11][0]) + float(kps[12][0])) / 2.0
        dx = shoulder_x - hip_x
        dy = shoulder_y - hip_y
        if abs(dy) > 1e-6:
            angle = abs(np.degrees(np.arctan2(abs(dx), abs(dy))))
            diag.torso_angle = round(angle, 1)
            if angle > TORSO_LEAN_WARNING_DEG and phase == "UP":
                diag.torso_lean_warning = True
                issues.append("torso_lean")
                penalty += 15.0

    if len(hip_x_history) >= 10 and body_width_px > 10:
        recent = np.array(hip_x_history[-30:], dtype=np.float64)
        drift = float(np.std(recent))
        diag.hip_x_drift_ratio = round(drift / body_width_px, 3)
        if drift > STABILITY_DRIFT_RATIO * body_width_px:
            diag.stability_warning = True
            issues.append("unstable")
            penalty += 10.0

    diag.issues = issues
    diag.score = round(max(0.0, 100.0 - penalty), 1)
    return diag


# ---------------------------------------------------------------------------
# 目标导向疲劳评估 (Goal-Oriented Velocity Loss)
# ---------------------------------------------------------------------------

MVT_STOP_THRESHOLD = 0.35  # 绝对速度低于此值直接 STOP
FATIGUE_THRESHOLDS = {"Power": 0.15, "Strength": 0.20, "Hypertrophy": 0.35}
WARNING_MARGIN = 0.05  # 阈值前 5% 为预警区间


@dataclass
class SetFatigueAssessment:
    """组内单次 Rep 疲劳评估结果"""
    status: str  # "STOP" | "Fatigued" | "Warning" | "Optimal"
    message: str
    velocity_loss_pct: float
    baseline_vel: float
    threshold_pct: float


def assess_set_fatigue(
    set_velocities: list[float],
    training_goal: str = "Strength",
) -> SetFatigueAssessment:
    """
    基于本组速度历史，评估当前 Rep 的疲劳程度。
    基准点: baseline_vel = max(set_velocities)
    绝对底线: current_velocity <= 0.35 m/s → STOP
    动态流失率: 根据 training_goal 使用不同阈值 (Power 15%, Strength 20%, Hypertrophy 35%)
    """
    threshold = FATIGUE_THRESHOLDS.get(training_goal, 0.20)
    threshold_pct = threshold * 100

    if not set_velocities:
        return SetFatigueAssessment(
            status="Optimal",
            message="✅ 状态良好，继续保持！",
            velocity_loss_pct=0.0,
            baseline_vel=0.0,
            threshold_pct=threshold_pct,
        )

    current_velocity = float(set_velocities[-1])
    baseline_vel = float(max(set_velocities))

    # 绝对安全底线拦截
    if current_velocity <= MVT_STOP_THRESHOLD:
        return SetFatigueAssessment(
            status="STOP",
            message="🚨 绝对速度过低，已触及力竭边缘，必须停止！",
            velocity_loss_pct=100.0 if baseline_vel > 0 else 0.0,
            baseline_vel=baseline_vel,
            threshold_pct=threshold_pct,
        )

    if baseline_vel <= 0:
        return SetFatigueAssessment(
            status="Optimal",
            message="✅ 状态良好，继续保持！",
            velocity_loss_pct=0.0,
            baseline_vel=baseline_vel,
            threshold_pct=threshold_pct,
        )

    velocity_loss = (baseline_vel - current_velocity) / baseline_vel
    velocity_loss_pct = velocity_loss * 100.0

    # 状态分级
    if velocity_loss >= threshold:
        return SetFatigueAssessment(
            status="Fatigued",
            message="⚠️ 速度流失达标，建议结束本组。",
            velocity_loss_pct=velocity_loss_pct,
            baseline_vel=baseline_vel,
            threshold_pct=threshold_pct,
        )
    if velocity_loss >= threshold - WARNING_MARGIN:
        return SetFatigueAssessment(
            status="Warning",
            message="⚡ 接近预警线，准备停组。",
            velocity_loss_pct=velocity_loss_pct,
            baseline_vel=baseline_vel,
            threshold_pct=threshold_pct,
        )
    return SetFatigueAssessment(
        status="Optimal",
        message="✅ 状态良好，继续保持！",
        velocity_loss_pct=velocity_loss_pct,
        baseline_vel=baseline_vel,
        threshold_pct=threshold_pct,
    )


# ---------------------------------------------------------------------------
# Fatigue-Predictor 疲劳趋势预测
# ---------------------------------------------------------------------------

FATIGUE_VELOCITY_FLOOR = 0.35
FATIGUE_LOSS_CEILING = 40.0


@dataclass
class FatiguePrediction:
    """组内疲劳预测结果"""
    predicted_next_v: float = 0.0
    predicted_loss_pct: float = 0.0
    confidence: float = 0.0
    failure_warning: bool = False
    predicted_velocities: list[float] = field(default_factory=list)
    message: str = ""


def predict_fatigue(
    rep_velocities: list[float],
    best_velocity: float,
) -> Optional[FatiguePrediction]:
    """
    基于组内已有 Rep 速度，线性外推预测后续速度衰减。
    至少需要 3 个 Rep 数据点才能预测。
    """
    n = len(rep_velocities)
    if n < 3 or best_velocity <= 0:
        return None

    x = np.arange(1, n + 1, dtype=np.float64)
    y = np.array(rep_velocities, dtype=np.float64)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if abs(ss_xx) < 1e-12:
        return None

    slope = float(ss_xy / ss_xx)
    intercept = float(y_mean - slope * x_mean)

    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r_sq = max(0.0, 1.0 - ss_res / ss_tot) if abs(ss_tot) > 1e-12 else 0.0
    confidence = min(1.0, r_sq * min(1.0, n / 6.0))

    future_reps = list(range(n + 1, n + 4))
    predicted_vs = [max(0.0, slope * r + intercept) for r in future_reps]
    next_v = predicted_vs[0]
    predicted_loss = (1.0 - next_v / best_velocity) * 100.0 if best_velocity > 0 else 0.0

    failure = next_v < FATIGUE_VELOCITY_FLOOR or predicted_loss > FATIGUE_LOSS_CEILING

    if failure:
        msg = f"预测第 {n+1} 把速度 {next_v:.3f} m/s，预计力竭，请注意保护"
    elif predicted_loss > 25:
        msg = f"预测第 {n+1} 把速度 {next_v:.3f} m/s，衰减趋势明显"
    else:
        msg = f"预测第 {n+1} 把速度 {next_v:.3f} m/s，状态可控"

    full_pred = [slope * r + intercept for r in range(1, n + 4)]

    return FatiguePrediction(
        predicted_next_v=round(next_v, 4),
        predicted_loss_pct=round(predicted_loss, 1),
        confidence=round(confidence, 2),
        failure_warning=failure,
        predicted_velocities=[round(v, 4) for v in full_pred],
        message=msg,
    )
