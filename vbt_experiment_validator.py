#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vbt_experiment_validator.py

论文第5章误差分析 - 独立 Streamlit 页面

零侵入设计:
  - 不修改任何主系统文件
  - 只读复用 vbt_cv_engine.process_squat_video
  - 独立 DataFrame，不写主系统 DB

运行:
    cd /home/kiki-pi/vbt_project
    streamlit run vbt_experiment_validator.py --server.port 8502
"""
from __future__ import annotations
import io, logging, os, sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
try:
    import streamlit as st
except ImportError:
    print("pip install streamlit"); sys.exit(1)

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

try:
    from vbt_cv_engine import process_squat_video
    _CV_OK = True; _CV_ERR = ""
except ImportError as _e:
    _CV_OK = False; _CV_ERR = str(_e)

try:
    from squat_analysis_core import ANATOMY_RATIO_SHOULDER_ANKLE
except ImportError:
    ANATOMY_RATIO_SHOULDER_ANKLE = 0.80

try:
    from vbt_analytics_pro import ensure_db_safe, DB_PATH
    _DB_OK = True
except ImportError:
    _DB_OK = False; DB_PATH = "squat_gym.db"

try:
    from vbt_runtime_config import get_user_height_cm as _dflt_height
except ImportError:
    def _dflt_height(): return 175.0

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vbt_exp_validator")

VIDEOS_DIR = os.path.join(_DIR, "videos")
UPLOAD_DIR = os.path.join(_DIR, "temp", "experiment_uploads")
EXP_KEY = "exp_val_df"
COLS = [
    "video_name", "saved_path", "subject_id", "user_height_cm", "camera_pitch_deg",
    "camera_to_subject_distance_cm",
    "camera_height_cm",
    "calib_method", "timing_source", "ref_target",
    "gt_ref_cm", "estimated_ref_cm", "estimate_source", "calib_error_pct",
    "is_placeholder", "reps_detected", "best_mcv_m_s", "mean_mcv_m_s",
    "analysis_status", "analysis_notes",
    "data_source", "is_demo",
]


# ---------------------------------------------------------------------------
# Core: wrap process_squat_video and extract experiment metrics
# ---------------------------------------------------------------------------

def run_analysis(
    video_path: str,
    user_height_cm: float,
    calib_method: str,
    plate_diameter_cm: float = 45.0,
    ph=None,
) -> dict:
    """Call process_squat_video (read-only reuse), extract experiment metrics.

    estimated_ref_cm source:
      shoulder_ankle -> height_cm * ANATOMY_RATIO_SHOULDER_ANKLE  (calibration math)
      fixed_scale    -> plate_diameter_cm                          (system input value)
      other          -> estimate_source = 'unavailable'
    """
    res = dict(
        estimated_ref_cm=None, estimate_source="unavailable",
        timing_source="video_fps",
        reps_detected=0, best_mcv_m_s=None, mean_mcv_m_s=None,
        analysis_status="error", analysis_notes="",
    )
    if not _CV_OK:
        res["analysis_notes"] = f"vbt_cv_engine unavailable: {_CV_ERR}"
        return res
    if not os.path.isfile(video_path):
        res["analysis_notes"] = f"File not found: {video_path}"
        return res
    if _DB_OK:
        try:
            ensure_db_safe(DB_PATH)
        except Exception as e:
            log.warning("ensure_db_safe non-fatal: %s", e)
    try:
        frames = 0
        last: dict = {}
        # ── 缓存最近一次非空标定诊断字段（防止 video_ended 帧丢失这些值）──
        last_nonnull_scale: Optional[float] = None
        last_nonnull_body_height_px: Optional[float] = None
        last_nonnull_method: Optional[str] = None
        last_nonnull_fallback: Optional[bool] = None
        gen = process_squat_video(
            video_source=video_path,
            user_height_cm=float(user_height_cm),
            use_plate_calibration=(calib_method == "fixed_scale"),
            plate_diameter_cm=float(plate_diameter_cm) if calib_method == "fixed_scale" else None,
            pose_diag_enabled=False,
            record_video=False,
        )
        if gen is None:
            res["analysis_notes"] = "process_squat_video returned None"
            return res
        for _f, stats in gen:
            frames += 1
            last = stats
            # 每帧更新非空缓存
            if stats.get("calib_scale_m_per_px") is not None:
                last_nonnull_scale = stats["calib_scale_m_per_px"]
            if stats.get("calib_body_height_px") is not None:
                last_nonnull_body_height_px = stats["calib_body_height_px"]
            if stats.get("calib_method_used") is not None:
                last_nonnull_method = stats["calib_method_used"]
            if stats.get("calib_is_fallback") is not None:
                last_nonnull_fallback = stats["calib_is_fallback"]
            if stats.get("video_ended"):
                break
            if ph and frames % 30 == 0:
                ph.info(f"Analyzing... {frames} frames | Reps: {stats.get('reps', 0)}")
        vels: list = last.get("rep_velocities", [])
        reps = int(last.get("reps", 0))
        best = last.get("best_vel") or None
        mean_v = float(np.mean(vels)) if vels else None

        # ── 优先使用缓存的非空标定字段（而非仅依赖最后一帧）──────────────────
        scale_m_per_px    = last_nonnull_scale
        body_height_px    = last_nonnull_body_height_px
        calib_method_used = last_nonnull_method
        calib_is_fallback = last_nonnull_fallback if last_nonnull_fallback is not None else False
        log.info(
            "[calib_diag] video=%s scale=%.6f body_px=%.1f method=%s fallback=%s",
            os.path.basename(video_path),
            scale_m_per_px or 0.0,
            body_height_px or 0.0,
            calib_method_used,
            calib_is_fallback,
        )

        # ── estimated_ref_cm：来自视频标定结果的独立估计，非真值回填 ─────────
        # shoulder_ankle:
        #   系统实际测量的肩踝像素距 = body_height_px * ANATOMY_RATIO_SHOULDER_ANKLE
        #   物理长度 = body_height_px * scale_m_per_px * 100  (cm)
        #   注：body_height_px 已在 CalibrationState 内部由实际视频关键点测量而来，
        #       与 gt_ref_cm（人工物理卷尺值）是完全独立的两个来源。
        #
        # fixed_scale:
        #   系统实际检测到的杠铃片像素直径 * scale = plate_diameter_used_m
        #   由于 scale 本身由杠铃片推导，此路径仍然循环依赖。
        #   更真实的方式是：用 shoulder_ankle 得到的 scale，反推杠铃片的 px 直径对应的物理长度。
        #   当前如果引擎未暴露杠铃片检测像素直径，标记为 unavailable。

        if scale_m_per_px is not None and scale_m_per_px > 0 and body_height_px is not None and body_height_px > 0:
            if calib_method == "shoulder_ankle":
                # 真实来源：视频中关键点实测的肩踝像素距 * 标定比例尺
                # body_height_px = shoulder_ankle_px / ANATOMY_RATIO_SHOULDER_ANKLE
                shoulder_ankle_px = body_height_px * ANATOMY_RATIO_SHOULDER_ANKLE
                estimated_m = shoulder_ankle_px * scale_m_per_px
                res["estimated_ref_cm"] = round(estimated_m * 100.0, 2)
                res["estimate_source"] = "video_keypoint_shoulder_ankle"
            elif calib_method == "fixed_scale":
                # fixed_scale 的 scale_m_per_px 本身由 plate_diameter_cm 推导，
                # 若用该 scale 反推 plate，会得到恒等结果（循环依赖）。
                # 此处改为：用 fixed_scale 得到的比例尺，推算视频中「身体参考长度」作为交叉验证
                # 即：估计的 shoulder_ankle 长度（用 fixed_scale 标定的 scale）
                shoulder_ankle_px = body_height_px * ANATOMY_RATIO_SHOULDER_ANKLE
                estimated_m = shoulder_ankle_px * scale_m_per_px
                res["estimated_ref_cm"] = round(estimated_m * 100.0, 2)
                res["estimate_source"] = "video_keypoint_via_fixed_scale"
                # 注：gt_ref_cm 此时应填肩踝真值，而非杠铃片直径，才能形成有效误差对比
            else:
                res["estimate_source"] = "unavailable"
        else:
            # 标定未完成或 scale 未获取，不伪造数值
            res["estimated_ref_cm"] = None
            res["estimate_source"] = "unavailable_calib_not_ready"

        notes_extra = (
            f" | calib={calib_method_used or 'unknown'}"
            f" | fallback={calib_is_fallback}"
            f" | scale={scale_m_per_px:.6f} m/px" if scale_m_per_px else ""
        )
        res.update(
            reps_detected=reps,
            best_mcv_m_s=round(float(best), 4) if best else None,
            mean_mcv_m_s=round(float(mean_v), 4) if mean_v else None,
            analysis_status="ok" if reps > 0 else "no_reps",
            analysis_notes=f"{frames} frames{notes_extra}",
        )
    except Exception as exc:
        res["analysis_notes"] = f"Exception: {exc}"
        res["analysis_status"] = "error"
        log.exception("Analysis error: %s", video_path)
    return res


def calib_error(gt, est) -> Optional[float]:
    try:
        g, e = float(gt), float(est)
        if abs(g) > 1e-6:
            return round((e - g) / g * 100.0, 3)
    except (TypeError, ValueError):
        pass
    return None


def _list_videos(d: str) -> list:
    if not os.path.isdir(d):
        return []
    return sorted(
        f for f in os.listdir(d)
        if Path(f).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
    )


def generate_demo_data() -> pd.DataFrame:
    """Generate synthetic demo rows for UI debugging and chart preview.
    All rows are tagged data_source='synthetic_demo' and is_demo=True.
    NOT for use in real paper statistics.
    """
    rng = np.random.default_rng(42)
    rows = []
    heights    = [165.0, 170.0, 175.0, 180.0]
    pitches    = [-5.0, 0.0, 5.0]
    distances  = [200.0, 300.0, 400.0]
    cam_heights = [60.0, 80.0, 100.0]
    methods    = ["shoulder_ankle", "fixed_scale"]
    timings    = ["video_fps", "wall_clock"]

    for i, (h, pitch, dist, cam_h, method, timing) in enumerate(
        [
            (h, p, d, ch, m, t)
            for h in heights
            for p in pitches
            for d in distances[:2]
            for ch in cam_heights[:2]
            for m in methods
            for t in timings[:1]   # keep demo set manageable
        ][:30]  # cap at 30 rows
    ):
        # shoulder_ankle: lower mean error (~2%), fixed_scale: higher (~4%)
        base_err = 2.0 if method == "shoulder_ankle" else 4.0
        noise = rng.uniform(-1.5, 1.5)
        err_pct = round(base_err + noise + abs(pitch) * 0.1 + (dist - 300) * 0.003, 3)
        gt = round(h * 0.80, 1) if method == "shoulder_ankle" else 45.0
        est = round(gt * (1 + err_pct / 100.0), 2)
        reps = int(rng.integers(4, 9))
        best_mcv = round(rng.uniform(0.55, 0.95), 3)
        mean_mcv = round(best_mcv * rng.uniform(0.80, 0.95), 3)
        rows.append({
            "video_name":   f"demo_{i+1:02d}_{method[:3]}.mp4",
            "saved_path":   "",
            "subject_id":   f"S{(i % 4) + 1:02d}",
            "user_height_cm": h,
            "camera_pitch_deg": pitch,
            "camera_to_subject_distance_cm": dist,
            "camera_height_cm": cam_h,
            "calib_method":  method,
            "timing_source": timing,
            "ref_target":    method,
            "gt_ref_cm":     gt,
            "estimated_ref_cm": est,
            "estimate_source": "synthetic_demo",
            "calib_error_pct": err_pct,
            "is_placeholder": False,
            "reps_detected": reps,
            "best_mcv_m_s":  best_mcv,
            "mean_mcv_m_s":  mean_mcv,
            "analysis_status": "ok",
            "analysis_notes": "synthetic demo row",
            "data_source":   "synthetic_demo",
            "is_demo":       True,
        })
    return pd.DataFrame(rows)


def _init():
    if EXP_KEY not in st.session_state:
        st.session_state[EXP_KEY] = pd.DataFrame(columns=COLS)


def _df() -> pd.DataFrame:
    return st.session_state[EXP_KEY].copy()


def _save(df: pd.DataFrame):
    st.session_state[EXP_KEY] = df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_CSS = """
<style>
[data-testid="stAppViewContainer"]{background:#0d1117;color:#cdd9e5;}
h1,h2,h3{color:#58a6ff;}
.stTabs [data-baseweb="tab"]{color:#8b949e;font-size:.95rem;}
.stTabs [aria-selected="true"]{color:#58a6ff;border-bottom:2px solid #58a6ff;}
.stDataFrame{border:1px solid #30363d;border-radius:6px;}
</style>
"""


def _render_tab1():
    st.subheader("实验全局默认参数")
    st.info("以下参数作为新增样本行的默认值，可在「视频样本」Tab 逐行覆盖。")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("被试身高 (cm)", 100.0, 220.0, float(_dflt_height()), 1.0, key="cfg_h")
        st.number_input("俯仰角 deg（仰拍为负）", -30.0, 30.0, 0.0, 0.5, key="cfg_pitch")
    with c2:
        st.number_input("杠铃片真实直径 (cm)", 10.0, 60.0, 45.0, 0.5, key="cfg_plate")
        st.selectbox("默认标定方法", ["shoulder_ankle", "fixed_scale"], key="cfg_calib")
    with c3:
        st.number_input(
            "gt_ref_cm（肩踝距 or 杠铃片直径真值）",
            0.0, 300.0, 0.0, 0.5, key="cfg_gt",
            help="shoulder_ankle: 肩关节到踝关节物理距离 cm; fixed_scale: 杠铃片实际直径 cm",
        )
        st.text_input("默认被试 ID", "S01", key="cfg_subj")
    st.markdown("---")
    st.markdown(
        "**ref_target 说明**\n"
        "- `shoulder_ankle`: gt_ref_cm = 肩关节到踝关节物理距离 (cm)\n"
        "- `fixed_scale`: gt_ref_cm = 杠铃片实际直径 (cm)\n"
        "- estimated_ref_cm 由系统自动推算（非人工填写）\n"
        "- calib_error_pct = (estimated - gt) / gt * 100 %"
    )
    st.markdown("---")
    st.subheader("演示数据（调试用）")
    st.warning(
        "演示数据仅用于 UI 调试、图表预览和导出测试，**不可用于正式论文统计**。"
        "生成后可在统计页签中通过过滤选项排除。"
    )
    col_demo1, col_demo2 = st.columns([2, 3])
    with col_demo1:
        if st.button("生成演示数据（synthetic demo）", key="gen_demo_btn"):
            demo_df = generate_demo_data()
            existing = _df()
            # Remove any old demo rows first to avoid duplicates
            if not existing.empty and "is_demo" in existing.columns:
                existing = existing[existing["is_demo"] != True]
            _save(pd.concat([existing, demo_df], ignore_index=True))
            st.success(f"已生成 {len(demo_df)} 条演示数据，请前往后续 Tab 查看。")
            st.rerun()
    with col_demo2:
        df_now = _df()
        if not df_now.empty and "is_demo" in df_now.columns:
            demo_count = int(df_now["is_demo"].sum())
            if demo_count > 0:
                st.info(f"当前样本表含 **{demo_count}** 条演示数据（标记为 is_demo=True）。")
                if st.button("清除全部演示数据", key="clear_demo_btn"):
                    _save(df_now[df_now["is_demo"] != True])
                    st.rerun()


def _render_tab2():
    """Upload-based sample management. Directory scan is removed."""
    st.subheader("② 视频样本管理（本地上传）")
    st.info(
        "上传视频后为每个视频单独配置参数，然后在「③ 自动分析」Tab 点击「开始分析」。"
        "分析结果累积保存，不覆盖已完成结果。"
    )

    # ── 文件上传 ─────────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "上传实验视频（支持多选）",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=True,
        key="upload_videos",
        help=f"视频将保存到 {UPLOAD_DIR}",
    )

    if uploaded_files:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        default_h     = float(st.session_state.get("cfg_h",     _dflt_height()))
        default_pitch = float(st.session_state.get("cfg_pitch", 0.0))
        default_calib = st.session_state.get("cfg_calib", "shoulder_ankle")
        default_gt    = float(st.session_state.get("cfg_gt",    0.0))
        default_subj  = st.session_state.get("cfg_subj", "S01")

        st.markdown("#### 为每个视频单独配置实验参数")
        st.caption("默认值来自「① 实验配置」Tab，可在此逐个覆盖。")

        for uf in uploaded_files:
            saved_path = os.path.join(UPLOAD_DIR, uf.name)
            df_cur = _df()
            already = (not df_cur.empty) and (df_cur["video_name"] == uf.name).any()

            label = ("✅ " if already else "🎬 ") + uf.name
            if already:
                label += "  （已在样本表中）"

            with st.expander(label, expanded=not already):
                if already:
                    st.info("该视频已在样本表中。若需重新配置，请先在下方删除该行。")
                    continue

                c1, c2, c3 = st.columns(3)
                with c1:
                    h = st.number_input(
                        "身高 cm", 100.0, 220.0, default_h, 1.0,
                        key=f"h_{uf.name}"
                    )
                    pitch = st.number_input(
                        "俯仰角 deg", -30.0, 30.0, default_pitch, 0.5,
                        key=f"pitch_{uf.name}"
                    )
                with c2:
                    calib_idx = 0 if default_calib == "shoulder_ankle" else 1
                    calib_m = st.selectbox(
                        "标定方法", ["shoulder_ankle", "fixed_scale"],
                        index=calib_idx, key=f"calib_{uf.name}"
                    )
                    timing_m = st.selectbox(
                        "时间轴来源", ["video_fps", "wall_clock"],
                        key=f"timing_{uf.name}"
                    )
                with c3:
                    gt = st.number_input(
                        "gt_ref_cm（真值）", 0.0, 300.0, default_gt, 0.5,
                        key=f"gt_{uf.name}",
                        help="shoulder_ankle→肩踝距; fixed_scale→杠铃片实际直径"
                    )
                    subj = st.text_input(
                        "被试 ID", default_subj, key=f"subj_{uf.name}"
                    )
                    dist_cm = st.number_input(
                        "相机到被试距离 (cm)", 0.0, 2000.0, 300.0, 10.0,
                        key=f"dist_{uf.name}",
                        help="拍摄时相机镜头到被试正面的水平距离（cm）"
                    )
                    cam_height_cm = st.number_input(
                        "相机离地高度 (cm)", 0.0, 500.0, 80.0, 5.0,
                        key=f"cam_h_{uf.name}",
                        help="相机镜头距地面的垂直高度（cm）"
                    )

                if st.button(f"添加「{uf.name}」到样本表", key=f"add_btn_{uf.name}"):
                    with open(saved_path, "wb") as fout:
                        fout.write(uf.getbuffer())
                    new_row = {
                        "video_name":     uf.name,
                        "saved_path":     saved_path,
                        "subject_id":     subj,
                        "user_height_cm": h,
                        "camera_pitch_deg": pitch,
                        "camera_to_subject_distance_cm": dist_cm,
                        "camera_height_cm": cam_height_cm,
                        "calib_method":   calib_m,
                        "timing_source":  timing_m,
                        "ref_target":     calib_m,
                        "gt_ref_cm":      gt,
                        "estimated_ref_cm": None,
                        "estimate_source":  "pending",
                        "calib_error_pct":  None,
                        "is_placeholder":   True,
                        "reps_detected":    None,
                        "best_mcv_m_s":     None,
                        "mean_mcv_m_s":     None,
                        "analysis_status":  "pending",
                        "analysis_notes":   "",
                    }
                    _save(pd.concat([_df(), pd.DataFrame([new_row])], ignore_index=True))
                    st.success(f"已添加「{uf.name}」，请前往「③ 自动分析」Tab 开始分析。")
                    st.rerun()

    # ── 当前样本表 ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 当前样本表")
    df_now = _df()
    if df_now.empty:
        st.info("样本表为空，请先上传视频并点击添加。")
    else:
        disp_cols = [c for c in [
            "video_name", "subject_id", "user_height_cm", "camera_pitch_deg",
            "calib_method", "timing_source", "gt_ref_cm", "analysis_status",
        ] if c in df_now.columns]
        st.dataframe(df_now[disp_cols], use_container_width=True, height=280)

        col_del, col_clr = st.columns([3, 1])
        with col_del:
            names = df_now["video_name"].tolist()
            del_name = st.selectbox("删除指定行", ["（不删除）"] + names, key="del_sel")
            if st.button("删除", disabled=(del_name == "（不删除）")):
                _save(df_now[df_now["video_name"] != del_name])
                st.success(f"已删除「{del_name}」")
                st.rerun()
        with col_clr:
            st.write("")
            if st.button("清空全部"):
                _save(pd.DataFrame(columns=COLS))
                st.rerun()


def _render_tab3():
    st.subheader("自动分析（复用原有 process_squat_video）")
    df_now = _df()
    if df_now.empty:
        st.info("请先在「视频样本」Tab 添加样本。")
        return

    pending_idx = df_now.index[df_now["analysis_status"] == "pending"].tolist()
    done_count = len(df_now) - len(pending_idx)
    st.write(f"待分析: **{len(pending_idx)}** 条 | 已完成: **{done_count}** 条")

    if not _CV_OK:
        st.error(f"vbt_cv_engine 不可用，无法运行分析。\n{_CV_ERR}")
    elif not pending_idx:
        st.success("所有样本已分析完毕。")
    else:
        if st.button("运行自动分析"):
            prog = st.progress(0)
            ph = st.empty()
            df_work = _df()
            for n, idx in enumerate(pending_idx):
                row = df_work.loc[idx]
                # 优先使用上传时保存的 saved_path，兼容旧目录扫描方式
                vpath = str(row.get("saved_path") or "")
                if not vpath or not os.path.isfile(vpath):
                    vpath = os.path.join(UPLOAD_DIR, str(row["video_name"]))
                ph.info(f"[{n+1}/{len(pending_idx)}] 分析: {row['video_name']}")
                r = run_analysis(
                    video_path=vpath,
                    user_height_cm=float(row["user_height_cm"]),
                    calib_method=str(row["calib_method"]),
                    plate_diameter_cm=float(st.session_state.get("cfg_plate", 45.0)),
                    ph=ph,
                )
                for k in ("estimated_ref_cm", "estimate_source", "timing_source",
                           "reps_detected", "best_mcv_m_s", "mean_mcv_m_s",
                           "analysis_status", "analysis_notes"):
                    df_work.at[idx, k] = r[k]
                df_work.at[idx, "is_placeholder"] = False
                df_work.at[idx, "calib_error_pct"] = calib_error(
                    row["gt_ref_cm"], r["estimated_ref_cm"]
                )
                prog.progress((n + 1) / len(pending_idx))
            _save(df_work)
            ph.success("分析完成！")
            st.rerun()

    st.dataframe(df_now, use_container_width=True, height=320)


def _render_tab4():
    st.subheader("标定误差验证")
    df_now = _df()
    # ── Demo 数据过滤 ────────────────────────────────────────────────────────
    has_demo = "is_demo" in df_now.columns and df_now["is_demo"].any()
    if has_demo:
        st.warning("当前数据包含演示数据（is_demo=True），默认已排除，不纳入正式统计。")
    include_demo = st.checkbox("纳入演示数据（仅调试用）", value=False, key="tab4_include_demo")
    if not include_demo and "is_demo" in df_now.columns:
        df_now = df_now[df_now["is_demo"] != True]
    done = df_now[df_now["analysis_status"].isin(["ok", "no_reps"])].copy()
    if done.empty:
        st.info("请先在「自动分析」Tab 运行分析后查看误差结果。")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # 图5.2  实验覆盖情况检查
    # ══════════════════════════════════════════════════════════════════════════
    # 有效样本：analysis_status==ok，estimated_ref_cm 和 calib_error_pct 非空
    valid_cov = done[
        (done["analysis_status"] == "ok") &
        done["estimated_ref_cm"].notna() &
        done["calib_error_pct"].notna()
    ].copy()

    st.markdown("---")
    st.markdown("### 图5.2 · 实验覆盖情况检查")
    st.caption(
        "统计当前有效样本（analysis_status=ok，estimated_ref_cm 和 calib_error_pct 非空）的条件覆盖情况，"
        "用于说明后续对比分析的实验条件是否充分。"
    )

    n_valid      = len(valid_cov)
    methods      = sorted(valid_cov["calib_method"].dropna().unique().tolist()) if n_valid else []
    timings      = sorted(valid_cov["timing_source"].dropna().unique().tolist()) if n_valid else []
    heights      = sorted(valid_cov["user_height_cm"].dropna().unique().tolist()) if n_valid else []
    pitches      = sorted(valid_cov["camera_pitch_deg"].dropna().unique().tolist()) if n_valid else []
    dists        = sorted(valid_cov["camera_to_subject_distance_cm"].dropna().unique().tolist()) \
                   if n_valid and "camera_to_subject_distance_cm" in valid_cov.columns else []
    cam_heights  = sorted(valid_cov["camera_height_cm"].dropna().unique().tolist()) \
                   if n_valid and "camera_height_cm" in valid_cov.columns else []

    def _fmt_list(lst, unit="") -> str:
        if not lst: return "—"
        s = ", ".join(str(v) for v in lst)
        return f"{s} {unit}".strip() if unit else s

    def _fmt_range(lst, unit="") -> str:
        if not lst: return "—"
        if len(lst) == 1: return f"{lst[0]} {unit}".strip()
        return f"{lst[0]}–{lst[-1]} {unit}".strip()

    # ── 第一行：核心数量指标 ────────────────────────────────────────────────
    ca, cb, cc, cd, ce = st.columns(5)
    ca.metric("有效样本总数",    str(n_valid))
    cb.metric("标定方法种数",    str(len(methods)))
    cc.metric("时间轴来源种数",  str(len(timings)))
    cd.metric("不同身高数",      str(len(heights)))
    ce.metric("不同俯仰角数",    str(len(pitches)))

    # ── 第二行：取值范围 ────────────────────────────────────────────────────
    st.markdown("")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.caption("**标定方法**")
    r1.write(_fmt_list(methods))
    r2.caption("**时间轴来源**")
    r2.write(_fmt_list(timings))
    r3.caption("**身高范围**")
    r3.write(_fmt_range(heights, "cm"))
    r4.caption("**俯仰角范围**")
    r4.write(_fmt_range(pitches, "°"))
    r5.caption("**相机距离范围**")
    r5.write(_fmt_range(dists, "cm") if dists else "—")

    if cam_heights:
        st.caption(f"相机离地高度范围：{_fmt_range(cam_heights, 'cm')}")

    # ── 覆盖结论自动判断 ────────────────────────────────────────────────────
    cov_ok = (
        n_valid > 0
        and len(methods) >= 2
        and len(timings) >= 1
        and len(heights) >= 2
        and len(pitches) >= 2
    )
    if n_valid == 0:
        st.info("暂无有效样本，请先完成视频分析。")
    elif cov_ok:
        st.success(
            "实验覆盖较完整，已包含多种标定方法、多个身高与俯仰角条件，"
            "可支持后续方法对比与误差分析。"
        )
    else:
        missing = []
        if len(methods) < 2:  missing.append("标定方法种数不足 2")
        if len(heights) < 2:  missing.append("身高条件不足 2 个")
        if len(pitches) < 2:  missing.append("俯仰角条件不足 2 个")
        st.warning(
            "实验覆盖仍有限，部分对照条件不足（" + "；".join(missing) + "），"
            "后续结论需结合实际条件谨慎解释。"
        )
    st.markdown("---")
    # ══════════════════════════════════════════════════════════════════════════

    # 展示误差统计
    valid = done.dropna(subset=["calib_error_pct"])
    if valid.empty:
        st.warning("所有行的 calib_error_pct 均为 None（gt_ref_cm 可能为 0 或 estimated_ref_cm 未获取）")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("平均误差 %", f"{valid['calib_error_pct'].mean():.3f} %")
        col2.metric("绝对误差均值 %", f"{valid['calib_error_pct'].abs().mean():.3f} %")
        col3.metric("最大绝对误差 %", f"{valid['calib_error_pct'].abs().max():.3f} %")

    st.markdown("#### 明细表")
    show_cols = ["video_name", "subject_id", "user_height_cm", "camera_pitch_deg",
                 "camera_to_subject_distance_cm", "camera_height_cm",
                 "calib_method", "gt_ref_cm", "estimated_ref_cm", "estimate_source",
                 "calib_error_pct", "reps_detected", "best_mcv_m_s"]
    st.dataframe(done[show_cols], use_container_width=True, height=320)

    st.markdown("#### 按标定方法分组统计")
    if "calib_method" in valid.columns:
        grp = valid.groupby("calib_method")["calib_error_pct"].agg(
            样本数="count", 均值="mean", 绝对均值=lambda x: x.abs().mean(), 标准差="std"
        ).reset_index()
        st.dataframe(grp, use_container_width=True)

    st.markdown("#### 按身高分组统计")
    if "user_height_cm" in valid.columns:
        grp_h = valid.groupby("user_height_cm")["calib_error_pct"].agg(
            样本数="count", 均值="mean", 绝对均值=lambda x: x.abs().mean()
        ).reset_index()
        st.dataframe(grp_h, use_container_width=True)

    st.markdown("#### 按俯仰角分组统计")
    if "camera_pitch_deg" in valid.columns:
        grp_p = valid.groupby("camera_pitch_deg")["calib_error_pct"].agg(
            样本数="count", 均值="mean", 绝对均值=lambda x: x.abs().mean()
        ).reset_index()
        st.dataframe(grp_p, use_container_width=True)


def _render_tab5():
    st.subheader("时间轴修正验证")
    df_now = _df()
    # ── Demo 数据过滤 ────────────────────────────────────────────────────────
    has_demo = "is_demo" in df_now.columns and df_now["is_demo"].any()
    if has_demo:
        st.warning("当前数据包含演示数据（is_demo=True），默认已排除，不纳入正式统计。")
    include_demo = st.checkbox("纳入演示数据（仅调试用）", value=False, key="tab5_include_demo")
    if not include_demo and "is_demo" in df_now.columns:
        df_now = df_now[df_now["is_demo"] != True]
    done = df_now[df_now["analysis_status"].isin(["ok", "no_reps"])].copy()
    if done.empty:
        st.info("请先完成自动分析。")
        return

    st.markdown(
        "当前所有离线视频分析统一使用 `video_fps` 时间轴。\n"
        "若实验视频含 wall_clock 对比数据，请在样本表中手动将 `timing_source` 标记为 `wall_clock`，"
        "并在此处比对 MCV 差异。"
    )

    show_cols = ["video_name", "calib_method", "timing_source",
                 "reps_detected", "best_mcv_m_s", "mean_mcv_m_s", "analysis_status"]
    st.dataframe(done[show_cols], use_container_width=True, height=320)

    if done["timing_source"].nunique() > 1:
        st.markdown("#### 按时间轴来源分组")
        grp = done.groupby("timing_source")[["best_mcv_m_s", "mean_mcv_m_s"]].mean().reset_index()
        st.dataframe(grp, use_container_width=True)
    else:
        st.info("当前所有样本均使用相同时间轴来源，无法分组比较。")


def _render_tab6():
    st.subheader("导出实验数据")
    df_now = _df()
    if df_now.empty:
        st.info("样本表为空，暂无数据可导出。")
        return

    col1, col2, col3 = st.columns(3)

    # CSV
    with col1:
        csv_buf = df_now.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="下载 CSV",
            data=csv_buf,
            file_name="vbt_experiment_results.csv",
            mime="text/csv",
        )

    # Excel
    with col2:
        xl_buf = io.BytesIO()
        with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
            df_now.to_excel(writer, index=False, sheet_name="实验结果")
            # 统计摘要 sheet
            valid = df_now.dropna(subset=["calib_error_pct"])
            if not valid.empty:
                summary = valid.groupby("calib_method")["calib_error_pct"].agg(
                    count="count", mean="mean",
                    abs_mean=lambda x: x.abs().mean(), std="std"
                ).reset_index()
                summary.to_excel(writer, index=False, sheet_name="误差摘要")
        st.download_button(
            label="下载 Excel",
            data=xl_buf.getvalue(),
            file_name="vbt_experiment_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # 文字摘要
    with col3:
        valid = df_now.dropna(subset=["calib_error_pct"])
        if not valid.empty:
            lines = [
                "=== VBT 实验误差摘要 ===",
                f"总样本数: {len(df_now)}",
                f"已分析: {len(valid)}",
                f"平均 calib_error_pct: {valid['calib_error_pct'].mean():.3f} %",
                f"绝对均值: {valid['calib_error_pct'].abs().mean():.3f} %",
                f"最大绝对误差: {valid['calib_error_pct'].abs().max():.3f} %",
                "",
                "=== 按标定方法 ===",
            ]
            for m, grp in valid.groupby("calib_method"):
                lines.append(
                    f"  {m}: n={len(grp)}, mean={grp['calib_error_pct'].mean():.3f}%, "
                    f"abs_mean={grp['calib_error_pct'].abs().mean():.3f}%"
                )
            summary_txt = "\n".join(lines)
            st.download_button(
                label="下载统计摘要",
                data=summary_txt.encode("utf-8"),
                file_name="vbt_experiment_summary.txt",
                mime="text/plain",
            )
        else:
            st.info("尚无有效误差数据可导出摘要。")

    st.markdown("---")
    st.markdown("#### 完整数据预览")
    st.dataframe(df_now, use_container_width=True, height=400)


def main():
    st.set_page_config(page_title="VBT 实验验证", page_icon="🔬", layout="wide")
    _init()
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title("🔬 VBT 实验验证 · 误差分析")
    st.caption("论文第5章 — 独立测试页面，主系统零影响")

    if not _CV_OK:
        st.error(f"vbt_cv_engine 导入失败，自动分析不可用。\n\n{_CV_ERR}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "① 实验配置", "② 视频样本", "③ 自动分析",
        "④ 标定误差", "⑤ 时间轴验证", "⑥ 导出",
    ])
    with tab1: _render_tab1()
    with tab2: _render_tab2()
    with tab3: _render_tab3()
    with tab4: _render_tab4()
    with tab5: _render_tab5()
    with tab6: _render_tab6()


if __name__ == "__main__":
    main()
