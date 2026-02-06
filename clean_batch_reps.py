#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗（可选）：删除“误判”动作——膝角大于 150°（根本没蹲下去）的 batch_reps 记录。
表字段为 min_knee_angle（非 knee_angle）。
"""

import sqlite3

DB_NAME = "squat_gym.db"

def main():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM batch_reps WHERE min_knee_angle IS NOT NULL AND min_knee_angle > 150")
    n = c.fetchone()[0]
    c.execute("DELETE FROM batch_reps WHERE min_knee_angle IS NOT NULL AND min_knee_angle > 150")
    conn.commit()
    conn.close()
    print(f"已删除 {n} 条膝角>150° 的无效记录（未真正蹲下）。")

if __name__ == "__main__":
    main()
