#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 session 名合并/去重 batch_reps：将同一 session 的多条 filename 记录合并为一条 canonical filename。
规则：去掉扩展名后，将全角括号、括号内数字等归一为同一 session_id（如 IMG_1696（1）.mov 与 IMG_1696.mov → IMG_1696），
再选一个规范 filename，把所有 rep 按原顺序合并并重新编号 rep_no。
"""

import os
import re
import sqlite3
from collections import defaultdict

DB_PATH = "squat_gym.db"
BATCH_TABLE = "batch_reps"


def normalize_session_id(filename):
    """
    将 filename 归一为 session_id，用于分组。
    例: IMG_1696（1）.mov -> IMG_1696,  IMG_1696.mov -> IMG_1696
    """
    base = os.path.splitext(filename)[0]
    # 去掉全角/半角括号及其中内容，如 （1） (1) （8）
    base = re.sub(r"[（(].*?[）)]", "", base)
    base = base.strip(" _")
    return base or filename


def choose_canonical_filename(filenames):
    """在同组多个 filename 中选一个作为规范名：优先无括号的短名，否则按字母序取第一个。"""
    def key(f):
        has_paren = "(" in f or "（" in f
        return (has_paren, f.lower())
    return min(filenames, key=key)


def merge_batch_reps_by_session(db_path=DB_PATH, dry_run=True):
    """
    按 session_id 合并 batch_reps：同 session 多 filename 合并为一个 canonical filename，rep_no 重排为 1..N。
    dry_run=True 时只打印将要执行的操作，不写库。
    返回 (groups_affected, total_rows_merged)。
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE batch_reps ADD COLUMN velocity_loss REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    cur.execute("""
        SELECT id, filename, rep_no, v_mean, min_knee_angle, max_trunk_angle, dtw_similarity, barbell_path_y, velocity_loss
        FROM batch_reps
        ORDER BY id
    """)
    rows = [dict(r) for r in cur.fetchall()]

    # 按归一化 session_id 分组
    by_session = defaultdict(list)
    for r in rows:
        sid = normalize_session_id(r["filename"])
        by_session[sid].append(r)

    # 只处理“同一 session 对应多个 filename”的组
    to_merge = {sid: group for sid, group in by_session.items() if len(set(r["filename"] for r in group)) > 1}
    if not to_merge:
        conn.close()
        print("无需合并：不存在同一 session 对应多个 filename 的情况。")
        return 0, 0

    total_rewritten = 0
    for session_id, group in sorted(to_merge.items()):
        filenames = list({r["filename"] for r in group})
        canonical = choose_canonical_filename(filenames)
        # 按原 id 顺序，合并后 rep_no 重排为 1, 2, ...
        ordered = sorted(group, key=lambda r: (r["id"], r["rep_no"]))
        if dry_run:
            print(f"[Dry-run] Session «{session_id}»: 合并 {len(ordered)} 条 (来自 {filenames}) -> 规范名 «{canonical}», rep_no 1..{len(ordered)}")
        else:
            # 删除该组所有涉及的 filename 的记录
            for f in filenames:
                cur.execute("DELETE FROM batch_reps WHERE filename = ?", (f,))
            # 按顺序重新插入，rep_no 从 1 开始
            for new_rep_no, r in enumerate(ordered, start=1):
                cur.execute("""
                    INSERT INTO batch_reps (filename, rep_no, v_mean, min_knee_angle, max_trunk_angle, dtw_similarity, barbell_path_y, velocity_loss)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    canonical,
                    new_rep_no,
                    r["v_mean"],
                    r["min_knee_angle"],
                    r["max_trunk_angle"],
                    r["dtw_similarity"],
                    r["barbell_path_y"],
                    r.get("velocity_loss"),
                ))
            total_rewritten += len(ordered)
            print(f"已合并 Session «{session_id}»: {len(ordered)} 条 -> «{canonical}», rep_no 1..{len(ordered)}")

    if not dry_run:
        conn.commit()
    conn.close()
    return len(to_merge), total_rewritten


def main():
    import argparse
    p = argparse.ArgumentParser(description="按 session 名合并 batch_reps，去重同一 session 的多 filename")
    p.add_argument("--apply", action="store_true", help="真正写库；默认仅 dry-run 打印")
    args = p.parse_args()

    if not os.path.isfile(DB_PATH):
        print(f"数据库不存在: {DB_PATH}")
        return

    dry_run = not args.apply
    if dry_run:
        print("当前为 dry-run 模式，仅打印将执行的操作。要写库请加 --apply\n")
    n_groups, n_rows = merge_batch_reps_by_session(DB_PATH, dry_run=dry_run)
    if n_groups > 0 and not dry_run:
        print(f"\n合并完成：{n_groups} 个 session，共 {n_rows} 条记录已合并为规范 filename。")


if __name__ == "__main__":
    main()
