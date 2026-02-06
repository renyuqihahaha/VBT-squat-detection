#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Mastery Dashboard — 教练视角的 VBT 综合分析
- 数据源: squat_gym.db (batch_reps + reps)
- 指标: MCV, 速度损失%, 速度变异系数 CV%, 动作一致性 (膝角/躯干角)
- 图表: 多线速度趋势图、每 session 速度损失柱状图
- AI 建议: 恢复、强度、中枢疲劳
[cite: 2026-01-24, 2026-01-25]
"""

import os
import sqlite3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

DB_PATH = "squat_gym.db"
REPORT_IMG = "vbt_pro_coach_report.png"


def _ascii_label(session_id, max_len=16):
    """缩短并统一为 ASCII 友好标签，避免全角括号等导致图表乱码。"""
    s = session_id.replace("（", "(").replace("）", ")").replace(" ", "")
    return (s[:max_len] + "..") if len(s) > max_len else s
HYPERTROPHY_VEL_MIN = 0.45
HYPERTROPHY_VEL_MAX = 0.65
TOO_LIGHT_THRESHOLD = 0.7
VEL_LOSS_RECOVERY_THRESHOLD = 30.0
READINESS_SLOW_THRESHOLD = 0.10  # 10% slower first rep = high central fatigue


def load_sessions_from_db():
    """
    从 batch_reps 和 reps 加载所有 session 数据。
    返回: list of dict:
      session_id, source, velocities, min_knee_angles, max_trunk_angles,
      rep_nos (或 rep_count), filename (batch) 或 ts (reps)
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    sessions = []

    # 1) batch_reps: 每个 filename 为一个 session
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='batch_reps'")
    if cur.fetchone():
        cur.execute("""
            SELECT filename, rep_no, v_mean, min_knee_angle, max_trunk_angle
            FROM batch_reps
            WHERE v_mean IS NOT NULL AND v_mean > 0
            ORDER BY filename, rep_no
        """)
        rows = cur.fetchall()
        by_file = {}
        for filename, rep_no, v_mean, min_knee, max_trunk in rows:
            if filename not in by_file:
                by_file[filename] = {"velocities": [], "min_knee": [], "max_trunk": [], "rep_nos": []}
            by_file[filename]["velocities"].append(float(v_mean))
            by_file[filename]["min_knee"].append(min_knee)
            by_file[filename]["max_trunk"].append(max_trunk)
            by_file[filename]["rep_nos"].append(rep_no)
        for filename, data in by_file.items():
            if data["velocities"]:
                sessions.append({
                    "session_id": os.path.splitext(filename)[0][:36],
                    "source": "batch",
                    "filename": filename,
                    "velocities": np.array(data["velocities"]),
                    "min_knee_angles": np.array([x for x in data["min_knee"] if x is not None], dtype=float) if any(x is not None for x in data["min_knee"]) else np.array([]),
                    "max_trunk_angles": np.array([x for x in data["max_trunk"] if x is not None], dtype=float) if any(x is not None for x in data["max_trunk"]) else np.array([]),
                    "rep_nos": data["rep_nos"],
                })

    # 2) reps: 按 ts 分组为 session（同一分钟内视为同一组）
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reps'")
    if cur.fetchone():
        cur.execute("""
            SELECT id, ts, rep_count, v_mean, left_knee_angle, right_knee_angle, trunk_angle
            FROM reps
            WHERE v_mean IS NOT NULL AND v_mean > 0
            ORDER BY id
        """)
        rows = cur.fetchall()
        if rows:
            # 简单按时间分组：相邻 60 秒内为同一 set
            groups = []
            current_ts = None
            current = []
            for r in rows:
                rid, ts, rep_count, v_mean, lk, rk, trunk = r
                t = datetime.fromisoformat(ts.replace("Z", "+00:00")) if "T" in ts else None
                ts_sec = t.timestamp() if t else rid
                if current_ts is None or (ts_sec - current_ts) < 120:
                    if current_ts is None:
                        current_ts = ts_sec
                    current.append((rep_count, float(v_mean), lk, rk, trunk))
                else:
                    if current:
                        groups.append((current_ts, current))
                    current_ts = ts_sec
                    current = [(rep_count, float(v_mean), lk, rk, trunk)]
            if current:
                groups.append((current_ts, current))
            for ts, recs in groups:
                velocities = [r[1] for r in recs]
                knee = []
                trunk = []
                for r in recs:
                    lk, rk, tr = r[2], r[3], r[4]
                    if lk is not None and rk is not None:
                        knee.append((float(lk) + float(rk)) / 2)
                    elif lk is not None:
                        knee.append(float(lk))
                    elif rk is not None:
                        knee.append(float(rk))
                    if tr is not None:
                        trunk.append(float(tr))
                sessions.append({
                    "session_id": f"realtime_{int(ts)}",
                    "source": "reps",
                    "filename": f"realtime_{int(ts)}",
                    "velocities": np.array(velocities),
                    "min_knee_angles": np.array(knee) if knee else np.array([]),
                    "max_trunk_angles": np.array(trunk) if trunk else np.array([]),
                    "rep_nos": list(range(1, len(recs) + 1)),
                })

    conn.close()
    return sessions


def compute_session_metrics(session):
    """单 session 指标: best_vel, avg_vel, velocity_loss_pct, cv_pct, form_consistency."""
    v = session["velocities"]
    if len(v) == 0:
        return None
    best_vel = float(np.max(v))
    avg_vel = float(np.mean(v))
    last_vel = float(v[-1])
    velocity_loss_pct = (best_vel - last_vel) / best_vel * 100 if best_vel > 0 else 0.0
    std_vel = np.std(v)
    cv_pct = (std_vel / avg_vel * 100) if avg_vel > 0 else 0.0

    # Form consistency from knee + trunk variability (lower CV = more consistent)
    knee = session.get("min_knee_angles")
    trunk = session.get("max_trunk_angles")
    if knee is None or (isinstance(knee, np.ndarray) and len(knee) == 0):
        knee = np.array([])
    else:
        knee = np.asarray(knee)
    if trunk is None or (isinstance(trunk, np.ndarray) and len(trunk) == 0):
        trunk = np.array([])
    else:
        trunk = np.asarray(trunk)
    form_cv = 0.0
    if len(knee) > 1:
        form_cv += np.std(knee) / (np.mean(knee) + 1e-6) * 100
    if len(trunk) > 1:
        form_cv += np.std(trunk) / (np.mean(trunk) + 1e-6) * 100
    if len(knee) > 1 or len(trunk) > 1:
        form_cv /= 2 if (len(knee) > 1 and len(trunk) > 1) else 1

    if form_cv < 5:
        form_label = "High"
    elif form_cv < 12:
        form_label = "Mid"
    else:
        form_label = "Low"

    return {
        "total_reps": len(v),
        "best_velocity": best_vel,
        "avg_velocity": avg_vel,
        "velocity_loss_pct": velocity_loss_pct,
        "cv_pct": cv_pct,
        "form_consistency": form_label,
        "form_cv": form_cv,
    }


def build_summary_table(sessions):
    """返回 (list of rows for table, list of metrics per session)."""
    rows = []
    metrics_list = []
    for s in sessions:
        m = compute_session_metrics(s)
        if m is None:
            continue
        metrics_list.append((s, m))
        sid = s["session_id"][:28] + ".." if len(s["session_id"]) > 30 else s["session_id"]
        rows.append([
            sid,
            m["total_reps"],
            f"{m['best_velocity']:.3f}",
            f"{m['avg_velocity']:.3f}",
            f"{m['velocity_loss_pct']:.1f}%",
            m["form_consistency"],
        ])
    return rows, metrics_list


def plot_trend_and_fatigue(sessions, metrics_list, save_path):
    """Graph 1: 多线速度趋势; Graph 2: 每 session 速度损失柱状图."""
    if not sessions or not metrics_list:
        # 空数据时生成占位图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.set_title("Velocity Trend (no data)")
        ax2.set_title("Velocity Loss % per Session (no data)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(sessions), 10)))

    # Graph 1: Multi-line velocity trend
    for i, (s, m) in enumerate(zip(sessions, [x[1] for x in metrics_list])):
        v = s["velocities"]
        x = np.arange(1, len(v) + 1)
        label = _ascii_label(s["session_id"], 20)
        ax1.plot(x, v, "o-", color=colors[i % len(colors)], label=label, markersize=4)
    ax1.set_xlabel("Rep number")
    ax1.set_ylabel("MCV (m/s)")
    ax1.set_title("Velocity Trend — Every Rep Across Sessions")
    ax1.legend(loc="best", fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Graph 2: Bar chart velocity loss %
    # 单 rep 的 session 速度为 0%，柱高为 0 会看不见，用最小显示高度并标注 [cite: 2026-02-06]
    labels = [_ascii_label(s["session_id"], 16) for s in sessions]
    loss_pcts = [m["velocity_loss_pct"] for _, m in metrics_list]
    min_bar_height = 1.0  # 至少显示 1% 高度，避免“数据缺失”观感
    display_heights = [max(p, min_bar_height) if p == 0 else p for p in loss_pcts]
    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, display_heights, color=colors[: len(labels)])
    # 在柱顶标注真实数值，单 rep 时标为 "0% (1rep)"
    for i, (bar, real_pct, (s, m)) in enumerate(zip(bars, loss_pcts, metrics_list)):
        h = display_heights[i]
        if real_pct == 0 and m["total_reps"] == 1:
            label = "0% (1rep)"
        else:
            label = f"{real_pct:.0f}%"
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.5, label, ha="center", va="bottom", fontsize=7, rotation=0)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Velocity Loss (%)")
    ax2.set_title("Velocity Loss % per Session (Fastest → Last Rep)")
    ax2.axhline(y=VEL_LOSS_RECOVERY_THRESHOLD, color="red", linestyle="--", alpha=0.7, label=f"Recovery threshold ({VEL_LOSS_RECOVERY_THRESHOLD}%)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def coaching_logic(metrics_list, sessions):
    """恢复建议、强度检查、中枢疲劳标记 → 返回建议列表."""
    suggestions = []
    if not metrics_list:
        return suggestions

    # Recovery: Velocity Loss > 30%
    for (s, m) in metrics_list:
        if m["velocity_loss_pct"] > VEL_LOSS_RECOVERY_THRESHOLD:
            suggestions.append({
                "type": "recovery",
                "session_id": s["session_id"],
                "message": f"Session «{s['session_id'][:24]}» 速度损失 {m['velocity_loss_pct']:.1f}% > {VEL_LOSS_RECOVERY_THRESHOLD}%。建议：增加组间休息或适当减轻负荷。",
            })
            break

    # Intensity: Best Velocity > 0.7 m/s → 当前重量偏轻，不利于肌肥大
    for (s, m) in metrics_list:
        if m["best_velocity"] > TOO_LIGHT_THRESHOLD:
            suggestions.append({
                "type": "intensity",
                "message": f"检测到最佳向心速度 {m['best_velocity']:.2f} m/s > {TOO_LIGHT_THRESHOLD} m/s。当前负荷偏轻，肌肥大区间建议控制在 {HYPERTROPHY_VEL_MIN}-{HYPERTROPHY_VEL_MAX} m/s [cite: 2026-01-25]。可考虑加重。",
            })
            break

    # Readiness: 最新一组的第一 rep 与之前 session 的第一 rep 比较，慢 10% → 中枢疲劳
    if len(sessions) >= 2 and len(metrics_list) >= 2:
        latest_first = sessions[-1]["velocities"][0] if len(sessions[-1]["velocities"]) else None
        prev_firsts = []
        for s in sessions[:-1]:
            if len(s["velocities"]):
                prev_firsts.append(s["velocities"][0])
        if latest_first is not None and prev_firsts:
            ref_first = float(np.mean(prev_firsts))
            if ref_first > 0 and (ref_first - latest_first) / ref_first >= READINESS_SLOW_THRESHOLD:
                suggestions.append({
                    "type": "readiness",
                    "message": f"最新一组首 rep 速度 ({latest_first:.3f} m/s) 较此前 session 平均首 rep ({ref_first:.3f} m/s) 下降 ≥10%。提示：High Central Fatigue，建议优先恢复再上强度。",
                })

    return suggestions


def print_markdown_table(rows):
    """终端输出 Markdown 表格."""
    if not rows:
        print("| Session ID | Total Reps | Best Velocity (m/s) | Average Velocity | Velocity Loss (%) | Form Consistency |")
        print("|------------|------------|---------------------|------------------|------------------|------------------|")
        print("| (no data)  | —          | —                   | —                | —                | —                |")
        return
    headers = ["Session ID", "Total Reps", "Best Velocity (m/s)", "Average Velocity", "Velocity Loss (%)", "Form Consistency"]
    col_widths = [max(len(str(h)), 14) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(c)))
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*r))


def main():
    print("=" * 60)
    print("  VBT Pro Coach Dashboard — Training Mastery Report")
    print("=" * 60)

    if not os.path.isfile(DB_PATH):
        print(f"数据库未找到: {DB_PATH}")
        print("请先运行 vbt_video_processor.py 或 vbt_analytics_pro.py 生成数据。")
        return

    sessions = load_sessions_from_db()
    if not sessions:
        print("未找到任何 session 数据（batch_reps / reps 为空或无可用的 v_mean）。")
        plot_trend_and_fatigue([], [], REPORT_IMG)
        print(f"已生成空报告图: {REPORT_IMG}")
        return

    rows, metrics_list = build_summary_table(sessions)
    # 只保留有 metrics 的 sessions 顺序（与 metrics_list 一致）
    sessions_with_metrics = [m[0] for m in metrics_list]

    # 1) 专业汇总表 (Markdown)
    print("\n## Session Summary Table\n")
    print_markdown_table(rows)

    # 2) 可视化
    plot_trend_and_fatigue(sessions_with_metrics, metrics_list, REPORT_IMG)
    print(f"\n📊 可视化报告已保存: {REPORT_IMG}")

    # 3) AI 教练建议
    suggestions = coaching_logic(metrics_list, sessions_with_metrics)
    print("\n## AI Coaching Notes\n")
    if not suggestions:
        print("✅ 未触发恢复/强度/疲劳预警，可维持当前计划。")
    else:
        for s in suggestions:
            print(f"• [{s['type'].upper()}] {s['message']}")

    # 4) 下次训练计划 (文本)
    print("\n## Next Session Plan (Suggested)\n")
    if metrics_list:
        last_m = metrics_list[-1][1]
        plan = []
        if any(s["type"] == "recovery" for s in suggestions):
            plan.append("• 增加组间休息 1–2 分钟，或本组减少 1–2 个 reps / 减轻 5–10% 负荷。")
        if any(s["type"] == "intensity" for s in suggestions):
            plan.append("• 适当加重，使向心速度落在 0.45–0.65 m/s 肌肥大区间。")
        if any(s["type"] == "readiness" for s in suggestions):
            plan.append("• 优先恢复：可做轻量技术组或延后大强度日。")
        if not plan:
            plan.append("• 保持当前负荷与组间休息，注意首 rep 速度与上一组一致性。")
        for p in plan:
            print(p)
    else:
        print("• 数据不足，建议先完成至少一组深蹲并录入后再生成计划。")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
