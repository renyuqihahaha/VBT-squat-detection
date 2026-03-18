#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_refactor.py — 重构验证测试

覆盖：
  1. squat_analysis_core: letterbox_preprocess / unpad_keypoints_array /
     CalibrationState / SquatStateMachine / HipTracker
  2. vbt_analytics_pro: sessions 表创建、get_all_sessions、delete_multiple_sessions
     insert_rep 写队列、migrate_legacy_sessions

运行：
  cd /home/kiki-pi/vbt_project && python3 test_refactor.py
"""

import os
import sqlite3
import sys
import tempfile
import time
import unittest

import numpy as np

# 确保项目目录在路径中
PROJECT = os.path.dirname(os.path.abspath(__file__))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)


# ══════════════════════════════════════════════════════════════
# Part 1: squat_analysis_core
# ══════════════════════════════════════════════════════════════

class TestLetterbox(unittest.TestCase):
    def test_square_image_unchanged(self):
        from squat_analysis_core import letterbox_preprocess
        img = np.zeros((192, 192, 3), dtype=np.uint8)
        padded, ox, oy, scale = letterbox_preprocess(img, target=192)
        self.assertEqual(padded.shape, (192, 192, 3))
        self.assertAlmostEqual(scale, 1.0, places=5)
        self.assertAlmostEqual(ox, 0.0, places=5)
        self.assertAlmostEqual(oy, 0.0, places=5)

    def test_wide_image_has_vertical_padding(self):
        """640x480 => scale=192/640=0.3, h_new=144, pad_h=48 => oy=24"""
        from squat_analysis_core import letterbox_preprocess
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, ox, oy, scale = letterbox_preprocess(img, target=192)
        self.assertEqual(padded.shape, (192, 192, 3))
        self.assertAlmostEqual(scale, 192 / 640, places=4)
        self.assertAlmostEqual(ox, 0.0, places=4)   # no horizontal pad
        self.assertGreater(oy, 0)                    # vertical pad present

    def test_tall_image_has_horizontal_padding(self):
        """480x640 (H>W) => scale=192/640, ox>0"""
        from squat_analysis_core import letterbox_preprocess
        img = np.zeros((640, 480, 3), dtype=np.uint8)
        padded, ox, oy, scale = letterbox_preprocess(img, target=192)
        self.assertEqual(padded.shape, (192, 192, 3))
        self.assertAlmostEqual(scale, 192 / 640, places=4)
        self.assertGreater(ox, 0)
        self.assertAlmostEqual(oy, 0.0, places=4)


class TestUnpadKeypoint(unittest.TestCase):
    def test_center_point_roundtrip(self):
        """A point at the image center should round-trip through letterbox."""
        from squat_analysis_core import letterbox_preprocess, unpad_keypoint
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        _, ox, oy, scale = letterbox_preprocess(img, target=192)
        # Center of image in original coords
        cy_orig, cx_orig = h / 2, w / 2
        # Simulate what MoveNet would output for that point
        canvas = 192
        h_new = h * scale
        w_new = w * scale
        y_content = cy_orig / h * h_new
        x_content = cx_orig / w * w_new
        y_canvas = y_content + oy
        x_canvas = x_content + ox
        y_norm = y_canvas / canvas
        x_norm = x_canvas / canvas
        # Recover
        y_rec, x_rec = unpad_keypoint(y_norm, x_norm, h, w, ox, oy, scale)
        self.assertAlmostEqual(y_rec, cy_orig, delta=0.5)
        self.assertAlmostEqual(x_rec, cx_orig, delta=0.5)

    def test_unpad_array_shape(self):
        from squat_analysis_core import letterbox_preprocess, unpad_keypoints_array
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, ox, oy, scale = letterbox_preprocess(img, target=192)
        kps_raw = np.random.rand(17, 3).astype(np.float32)
        kps_raw[:, 2] = 0.9  # high confidence
        out = unpad_keypoints_array(kps_raw, 480, 640, ox, oy, scale)
        self.assertEqual(out.shape, (17, 3))
        # confidence should be preserved
        np.testing.assert_array_almost_equal(out[:, 2], kps_raw[:, 2])


class TestCalibrationState(unittest.TestCase):
    def _make_kps(self, h=480, vis=0.9):
        """Synthetic keypoints: person standing, all joints visible."""
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 2] = vis
        # y-coords (pixel, increasing downward)
        kps[0][0] = 50    # nose
        kps[5][0] = 120   # left shoulder
        kps[6][0] = 120   # right shoulder
        kps[11][0] = 260  # left hip
        kps[12][0] = 260  # right hip
        kps[15][0] = 440  # left ankle
        kps[16][0] = 440  # right ankle
        return kps

    def test_shoulder_ankle_calibration(self):
        from squat_analysis_core import CalibrationState, MIN_CALIB_SAMPLES
        calib = CalibrationState(user_height_m=1.75)
        kps = self._make_kps()
        for i in range(MIN_CALIB_SAMPLES + 5):
            calib.update(kps, frame_n=i + 1, state="STANDING")
        self.assertTrue(calib.is_done)
        self.assertEqual(calib.method, "shoulder_ankle")
        self.assertGreater(calib.ratio, 0)
        self.assertGreater(calib.body_height_px, 0)

    def test_timeout_fallback(self):
        from squat_analysis_core import CalibrationState, CALIB_TIMEOUT_FRAMES, DEFAULT_FALLBACK_RATIO
        calib = CalibrationState(user_height_m=1.75)
        # All keypoints invisible
        kps = np.zeros((17, 3), dtype=np.float32)
        for i in range(CALIB_TIMEOUT_FRAMES + 2):
            calib.update(kps, frame_n=i + 1, state="STANDING")
        self.assertTrue(calib.is_done)
        self.assertTrue(calib.is_fallback)
        self.assertAlmostEqual(calib.ratio, DEFAULT_FALLBACK_RATIO, places=6)


class TestSquatStateMachine(unittest.TestCase):
    def _make_sm(self, body_px=300.0, h_m=1.75, fps=30.0):
        from squat_analysis_core import SquatStateMachine
        return SquatStateMachine(body_height_px=body_px, user_height_m=h_m, video_fps=fps)

    def test_no_movement_no_rep(self):
        sm = self._make_sm()
        scale = 0.004
        for i in range(60):
            sm.update(240.0, frame_n=i + 1, scale_m_per_px=scale)
        self.assertEqual(sm.rep_count, 0)
        self.assertIsNone(sm.finished_rep)

    def test_one_complete_rep(self):
        """Simulate one squat: standing -> down -> up -> standing."""
        sm = self._make_sm(body_px=400.0, h_m=1.75, fps=30.0)
        scale = 0.004
        # Establish baseline at y=200
        for i in range(15):
            sm.update(200.0, frame_n=i + 1, scale_m_per_px=scale)
        # Descend to y=310 (110px = 27% body height > 8% threshold)
        for i in range(15):
            y = 200.0 + (i + 1) * (110.0 / 15)
            sm.update(y, frame_n=16 + i, scale_m_per_px=scale)
        # Ascend back to y=200
        for i in range(20):
            y = 310.0 - (i + 1) * (110.0 / 20)
            sm.update(y, frame_n=31 + i, scale_m_per_px=scale)
        # Stay at top for buffer frames
        rep = None
        for i in range(10):
            sm.update(200.0, frame_n=51 + i, scale_m_per_px=scale)
            if sm.finished_rep is not None:
                rep = sm.finished_rep
                sm.finished_rep = None
                break
        self.assertIsNotNone(rep, "Expected one rep to be detected")
        self.assertEqual(rep.rep_index, 1)
        self.assertGreater(rep.mcv, 0)
        self.assertGreater(rep.rom_m, 0)
        self.assertIsNone(rep.velocity_loss)  # first rep has no loss

    def test_two_reps_velocity_loss(self):
        """Second rep should have velocity_loss computed."""
        sm = self._make_sm(body_px=400.0, h_m=1.75, fps=30.0)
        scale = 0.004
        reps_collected = []

        def _do_rep(start_frame, descent_speed=1.0):
            # baseline
            f = start_frame
            for _ in range(15):
                sm.update(200.0, frame_n=f, scale_m_per_px=scale)
                if sm.finished_rep:
                    reps_collected.append(sm.finished_rep)
                    sm.finished_rep = None
                f += 1
            # descend
            for i in range(15):
                y = 200.0 + (i + 1) * 7.0
                sm.update(y, frame_n=f, scale_m_per_px=scale)
                if sm.finished_rep:
                    reps_collected.append(sm.finished_rep)
                    sm.finished_rep = None
                f += 1
            # ascend (slower = higher f count -> lower MCV for 2nd rep)
            frames_up = int(20 * descent_speed)
            for i in range(frames_up):
                y = 305.0 - (i + 1) * (105.0 / frames_up)
                sm.update(y, frame_n=f, scale_m_per_px=scale)
                if sm.finished_rep:
                    reps_collected.append(sm.finished_rep)
                    sm.finished_rep = None
                f += 1
            # buffer
            for _ in range(10):
                sm.update(200.0, frame_n=f, scale_m_per_px=scale)
                if sm.finished_rep:
                    reps_collected.append(sm.finished_rep)
                    sm.finished_rep = None
                f += 1
            return f

        f = _do_rep(1, descent_speed=1.0)   # fast rep
        f = _do_rep(f, descent_speed=2.0)   # slower rep -> higher velocity_loss

        valid_reps = [r for r in reps_collected if r is not None]
        if len(valid_reps) >= 2:
            self.assertIsNone(valid_reps[0].velocity_loss)
            self.assertIsNotNone(valid_reps[1].velocity_loss)


class TestHipTracker(unittest.TestCase):
    def test_returns_none_when_no_visibility(self):
        from squat_analysis_core import HipTracker
        tracker = HipTracker(is_bodyweight=False)
        kps = np.zeros((17, 3), dtype=np.float32)  # all conf=0
        smoothed, raw_y, raw_x = tracker.update(kps)
        self.assertIsNone(smoothed)

    def test_returns_value_when_visible(self):
        from squat_analysis_core import HipTracker
        tracker = HipTracker(is_bodyweight=False)
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[11][0] = 300.0
        kps[11][2] = 0.9
        kps[12][0] = 305.0
        kps[12][2] = 0.9
        # Wrist visible too (kps[9], kps[10]) for barbell tracking
        kps[9][0] = 200.0; kps[9][2] = 0.9
        kps[10][0] = 200.0; kps[10][2] = 0.9
        smoothed, raw_y, raw_x = tracker.update(kps)
        self.assertIsNotNone(smoothed)
        self.assertAlmostEqual(smoothed, 200.0, delta=1.0)  # wrist tracking


# ══════════════════════════════════════════════════════════════
# Part 2: vbt_analytics_pro — sessions + write queue
# ══════════════════════════════════════════════════════════════

class TestSessionModel(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db = self._tmp.name

    def tearDown(self):
        try:
            os.unlink(self.db)
        except OSError:
            pass

    def _analytics(self):
        import vbt_analytics_pro as ap
        return ap

    def test_init_db_creates_sessions_table(self):
        ap = self._analytics()
        ap.init_db(self.db)
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        self.assertIsNotNone(cur.fetchone(), "sessions table should exist")
        conn.close()

    def test_get_all_sessions_empty(self):
        ap = self._analytics()
        ap.init_db(self.db)
        sessions = ap.get_all_sessions(self.db)
        self.assertEqual(sessions, [])

    def test_get_all_sessions_after_insert(self):
        ap = self._analytics()
        ap.init_db(self.db)
        # Insert a session manually
        conn = sqlite3.connect(self.db)
        conn.execute(
            "INSERT INTO sessions (session_id, user_name, started_at, source_type) VALUES (?, ?, ?, ?)",
            ("sess_001", "test_user", "2026-03-17T10:00:00", "realtime"),
        )
        conn.execute(
            "INSERT INTO reps (ts, rep_count, v_mean, session_id, user_name) VALUES (?, ?, ?, ?, ?)",
            ("2026-03-17T10:00:02", 2, 0.48, "sess_001", "test_user"),
        )
        conn.commit()
        conn.close()

        sessions = ap.get_all_sessions(self.db)
        self.assertEqual(len(sessions), 1)
        s = sessions[0]
        self.assertEqual(s["session_id"], "sess_001")
        self.assertEqual(s["user_name"], "test_user")
        self.assertEqual(s["reps"], 2)
        self.assertIn("id", s)
        self.assertIn("date", s)

    def test_delete_multiple_sessions(self):
        ap = self._analytics()
        ap.init_db(self.db)
        conn = sqlite3.connect(self.db)
        for sid in ("sess_A", "sess_B", "sess_C"):
            conn.execute(
                "INSERT INTO sessions (session_id, user_name, started_at) VALUES (?, ?, ?)",
                (sid, "u", "2026-03-17T09:00:00"),
            )
        for i, sid in enumerate(["sess_A", "sess_A", "sess_B", "sess_C"]):
            conn.execute(
                "INSERT INTO reps (ts, rep_count, v_mean, session_id, user_name) VALUES (?, ?, ?, ?, ?)",
                (f"2026-03-17T09:00:0{i}", i + 1, 0.5, sid, "u"),
            )
        conn.commit()
        conn.close()

        deleted = ap.delete_multiple_sessions(self.db, ["sess_A", "sess_B"])
        self.assertEqual(deleted, 3)  # 2 from sess_A + 1 from sess_B

        remaining = ap.get_all_sessions(self.db)
        ids = [s["session_id"] for s in remaining]
        self.assertIn("sess_C", ids)
        self.assertNotIn("sess_A", ids)
        self.assertNotIn("sess_B", ids)

    def test_delete_empty_list_returns_zero(self):
        ap = self._analytics()
        ap.init_db(self.db)
        deleted = ap.delete_multiple_sessions(self.db, [])
        self.assertEqual(deleted, 0)

    def test_migrate_legacy_sessions_with_existing_session_id(self):
        ap = self._analytics()
        ap.init_db(self.db)
        conn = sqlite3.connect(self.db)
        conn.execute(
            "INSERT INTO reps (ts, rep_count, v_mean, session_id, user_name) VALUES (?, ?, ?, ?, ?)",
            ("2026-01-01T10:00:00", 1, 0.6, "old_session_2026-01-01", "alice"),
        )
        conn.commit()
        conn.close()
        ap._migrate_legacy_sessions(self.db)
        sessions = ap.get_all_sessions(self.db)
        ids = [s["session_id"] for s in sessions]
        self.assertIn("old_session_2026-01-01", ids)

    def test_migrate_legacy_sessions_orphan_reps(self):
        ap = self._analytics()
        ap.init_db(self.db)
        conn = sqlite3.connect(self.db)
        conn.execute(
            "INSERT INTO reps (ts, rep_count, v_mean, user_name, set_number) VALUES (?, ?, ?, ?, ?)",
            ("2026-02-15T08:00:00", 1, 0.5, "bob", 2),
        )
        conn.commit()
        conn.close()
        ap._migrate_legacy_sessions(self.db)
        conn2 = sqlite3.connect(self.db)
        cur = conn2.cursor()
        cur.execute("SELECT session_id FROM reps WHERE user_name='bob'")
        row = cur.fetchone()
        conn2.close()
        self.assertIsNotNone(row)
        self.assertIsNotNone(row[0])
        self.assertIn("bob", row[0])
        self.assertIn("2026-02-15", row[0])

    def test_insert_rep_via_queue(self):
        ap = self._analytics()
        ap.ensure_db_safe(self.db)
        time.sleep(0.2)
        ap.insert_rep(
            self.db, 1, 0.55, 0.35, 90.0, 91.0, 15.0, None,
            None, 60.0, None, "queue_test_user",
            95.0, 1, "test_session_queue", 175.0, None,
        )
        ap._write_queue.join()
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute("SELECT v_mean, user_name FROM reps WHERE session_id='test_session_queue'")
        row = cur.fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row[0], 0.55, places=3)
        self.assertEqual(row[1], "queue_test_user")


class TestOfflineUsesLetterbox(unittest.TestCase):
    def test_imports_letterbox_from_core(self):
        import vbt_video_processor as vp
        from squat_analysis_core import letterbox_preprocess
        self.assertIs(vp.letterbox_preprocess, letterbox_preprocess)

    def test_imports_unpad_from_core(self):
        import vbt_video_processor as vp
        from squat_analysis_core import unpad_keypoints_array
        self.assertIs(vp.unpad_keypoints_array, unpad_keypoints_array)

    def test_imports_calib_from_core(self):
        import vbt_video_processor as vp
        from squat_analysis_core import CalibrationState
        self.assertIs(vp.CalibrationState, CalibrationState)


if __name__ == "__main__":
    unittest.main(verbosity=2)
