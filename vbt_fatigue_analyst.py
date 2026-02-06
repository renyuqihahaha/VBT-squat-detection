#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBT 疲劳分析：基于速度衰减率，不纠结“角度对不对”，计算每组动作的速度衰减并给出停止建议。
含数据平滑（插值）处理，应对丢帧/0° 等异常值，确保速度曲线平滑 [cite: 2026-01-24]。
"""

import sqlite3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# 数据库路径
DB_NAME = "squat_gym.db"


def load_velocities_from_db():
    """
    优先从 batch_reps 取最近一次训练（同一视频）的 v_mean；
    若无则从 reps 取最近一组；若无则返回 (None, None)。
    返回 (velocities_list, source_description)。
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # 1) 尝试 batch_reps：最近一次处理的视频，且过滤噪音 [cite: 2026-01-24]
    try:
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='batch_reps'")
        if c.fetchone():
            c.execute("SELECT filename FROM batch_reps ORDER BY id DESC LIMIT 1")
            row_name = c.fetchone()
            if row_name:
                filename = row_name[0]
                # 过滤：v_mean > 0.15 m/s（排除极慢干扰），min_knee_angle < 140°（确保真的蹲下去）
                c.execute("""
                    SELECT rep_no, v_mean FROM batch_reps
                    WHERE filename = ?
                    AND v_mean > 0.15
                    AND (min_knee_angle IS NULL OR min_knee_angle < 140)
                    ORDER BY rep_no ASC
                """, (filename,))
                rows = c.fetchall()
                if rows:
                    conn.close()
                    return [r[1] for r in rows], f"批处理视频: {filename}"
    except Exception:
        pass
    # 2) 尝试 reps：最近 N 条（实时/单视频录入）
    try:
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reps'")
        if c.fetchone():
            c.execute("SELECT ts, rep_count, v_mean FROM reps ORDER BY id DESC LIMIT 50")
            rows = c.fetchall()
            if rows:
                ts = rows[0][0] if rows else ""
                conn.close()
                return [r[2] for r in reversed(rows)], f"实时/单视频录入 (最近记录时间: {ts})"
    except Exception:
        pass
    conn.close()
    return None, None


def smooth_velocities(velocities):
    """
    数据平滑：将无效值（≤0、过小噪声、NaN）插值补齐，确保速度曲线可用 [cite: 2026-01-24]。
    将 0.0x 等明显噪声（如 < 0.05 m/s）视为无效并线性插值。
    """
    arr = np.asarray(velocities, dtype=float)
    s = pd.Series(arr)
    # 无效值：≤0、NaN、以及过小噪声（如 0.01/0.02 m/s，视为误判/站位调整）
    s[(s <= 0) | np.isnan(s) | (s < 0.05)] = np.nan
    if s.isna().all():
        return list(arr)
    s = s.interpolate(method="linear", limit_direction="both")
    s = s.ffill().bfill()
    return s.tolist()


def analyze_velocity_loss(threshold_percent=20.0):
    # 优先从数据库读取最近一次训练的速度
    reps_velocity, data_source = load_velocities_from_db()
    if not reps_velocity or len(reps_velocity) < 2:
        # 演示用：典型力竭深蹲 8 个 Rep 的向心速度 (m/s)
        reps_velocity = [0.45, 0.46, 0.44, 0.41, 0.38, 0.32, 0.25, 0.18]
        data_source = None
        print("📌 未从数据库读到足够数据，使用演示数据。")
        print("📂 目前分析的是：无（未关联任何视频，仅为演示曲线）")
        print("💡 若需分析真实视频：请先运行 python vbt_video_processor.py 处理 videos/ 下视频，或 vbt_analytics_pro.py 实时录入，再运行本脚本。")
    else:
        print(f"📂 目前分析的是：{data_source}")

    # 数据平滑（插值），应对丢帧/0° 等异常 [cite: 2026-01-24]
    reps_velocity = smooth_velocities(reps_velocity)
    n_reps = len(reps_velocity)

    print(f"📊 分析最新一组训练数据 (共 {n_reps} 次动作)...")

    # 1. 基准速度：取前两个动作的平均值
    baseline_vel = np.mean(reps_velocity[:2])
    print(f"🔹 基准速度 (Best Performance): {baseline_vel:.2f} m/s")

    cutoff_rep = None
    velocities = []
    losses = []

    print("\n--- 逐次动作分析 ---")
    for i, vel in enumerate(reps_velocity):
        rep_num = i + 1
        loss_pct = (baseline_vel - vel) / baseline_vel * 100 if baseline_vel > 0 else 0
        velocities.append(vel)
        losses.append(loss_pct)

        status = "✅ 状态良好"
        if loss_pct > 40:
            status = "❌ 严重力竭 (危险)"
        elif loss_pct > threshold_percent:
            status = "⚠️ 建议停止 (达到阈值)"
            if cutoff_rep is None:
                cutoff_rep = rep_num

        print(f"Rep {rep_num}: 速度 {vel:.2f} m/s | 衰减 {loss_pct:+.1f}% | {status}")

    # 2. 反馈报告
    print("\n📢 VBT 智能决策报告:")
    if cutoff_rep:
        print(f"🔴 系统建议：你应该在第 【{cutoff_rep}】 个动作后停止！")
        print(f"   原因：速度衰减超过 {threshold_percent}%，继续训练可能导致技术变形或无效疲劳。")
        print(f"   实际多做了 {n_reps - cutoff_rep} 个无效动作。")
    else:
        print("🟢 系统评价：完美！整组动作质量维持在预设阈值内。")

    # 3. 可视化
    plt.figure(figsize=(10, 6))
    x_reps = range(1, n_reps + 1)
    plt.plot(x_reps, reps_velocity, marker="o", label="Concentric Velocity", color="blue")
    cutoff_vel = baseline_vel * (1 - threshold_percent / 100)
    plt.axhline(y=cutoff_vel, color="red", linestyle="--", label=f"Cutoff Threshold (-{threshold_percent}%)")
    plt.title(f"Velocity Loss Analysis (Baseline: {baseline_vel:.2f} m/s)")
    plt.xlabel("Repetition Number")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True)
    plt.savefig("velocity_loss_report.png")
    print("📈 图表已生成: velocity_loss_report.png")


if __name__ == "__main__":
    analyze_velocity_loss(threshold_percent=25.0)  # 设定 2
