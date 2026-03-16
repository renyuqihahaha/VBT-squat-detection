#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBT 主入口：支持摄像头与本地视频分析。
通过 --source 指定视频源：摄像头索引 (0) 或 .mp4 文件路径。
集成疲劳预警 UI：当速度下降至 70% 以下时实时渲染红色警告。
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Union

import cv2

import physics_converter  # noqa: F401 - 确保物理换算模块加载
import vbt_analytics_pro  # noqa: F401 - 确保分析模块加载
from vbt_fatigue_analyst import is_fatigue_70_realtime

logger = logging.getLogger("vbt_main")


def _parse_source(source: str) -> Union[int, str]:
    """
    解析 --source 参数：.mp4 文件路径或摄像头索引。
    Returns:
        int: 摄像头索引 (0, 1, ...)
        str: 视频文件路径
    """
    if source.lower().endswith(".mp4") or source.lower().endswith(".mov"):
        if os.path.isfile(source):
            return source
        # 尝试相对于脚本目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(script_dir, source)
        if os.path.isfile(alt):
            return alt
        return source  # 原样传入，由 VideoCapture 报错
    try:
        return int(source)
    except ValueError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="VBT 深蹲分析：摄像头或本地视频")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="视频源：摄像头索引 (0) 或 .mp4/.mov 文件路径，如 test_squat.mp4",
    )
    args = parser.parse_args()

    video_source = _parse_source(args.source)
    is_file = isinstance(video_source, str)

    if is_file:
        logger.info("本地视频模式: %s", video_source)
    else:
        logger.info("摄像头模式: 索引 %d", video_source)

    try:
        from vbt_cv_engine import process_squat_video
        from vbt_runtime_config import get_user_height_cm
    except ImportError as e:
        logger.error("无法导入 vbt_cv_engine: %s", e)
        return

    gen = process_squat_video(video_source, user_height_cm=float(get_user_height_cm()))
    if gen is None:
        logger.error("无法打开视频源: %s", video_source)
        return

    window_name = "VBT 深蹲分析"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        for frame_bgr, stats in gen:
            v_base = stats.get("best_vel", 0.0)
            v_current = stats.get("current_vel", 0.0)
            is_fatigue = is_fatigue_70_realtime(v_base, v_current)

            if is_fatigue:
                h, w = frame_bgr.shape[:2]
                txt = "!! FATIGUE WARNING: 70%% reached - STOP !!"
                cv2.putText(
                    frame_bgr,
                    txt,
                    (max(20, w // 2 - 280), h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )

            cv2.imshow(window_name, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if stats.get("video_ended"):
                cv2.waitKey(2000)
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
