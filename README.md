# VBT-squat-detection

基于速度的深蹲监测系统（Velocity Based Training, VBT），运行在树莓派上的实时视觉算法。
通过测量深蹲向心阶段的平均速度（MCV），结合 AI 疲劳预测与训练模式策略，帮助你科学控制训练强度与疲劳水平。

---

## 技术栈

- **语言**: Python 3.9+
- **视觉/姿态**: OpenCV + MoveNet Lightning (TFLite)
- **数值计算**: NumPy (纯 numpy 推理，无需 TensorFlow 用于 inference)
- **数据库**: SQLite (WAL 模式，单写线程队列)
- **看板**: Streamlit

---

## 模块说明

| 文件 | 说明 |
|---|---|
| `squat_analysis_core.py` | 核心共享库：CalibrationState、SquatStateMachine、HipTracker |
| `vbt_cv_engine.py` | 实时 CV 引擎：摄像头/视频统一处理，生成逐帧分析结果 |
| `vbt_analytics_pro.py` | 数据库层：schema 管理、写队列、sessions、ML 表 |
| `vbt_training_modes.py` | **训练模式策略**：Power/Strength/Hypertrophy、SetLifecycleManager、Quality Gate、LoadRecommendation、SessionReport |
| `vbt_dl_models.py` | **DL 模块**：SetFatigueNet + TechniqueAnomalyNet（numpy 推理 + 规则回退）|
| `vbt_ml_pipeline.py` | **ML 流水线**：数据集构建、训练、评估、导出 |
| `vbt_pro_coach_dashboard.py` | Streamlit 看板：训练模式卡片、AI 输出、组间推荐时间线 |
| `vbt_ai_advisor.py` | LVP 负荷-速度建模、1RM 预测、训练建议 |
| `physics_converter.py` | 像素→米换算、Plane-to-Plane 深度偏移（无隐藏偏置）|
| `vbt_video_processor.py` | 离线批处理视频 |
| `run_tests.py` | 验证脚本（无框架依赖，43 项测试）|

---

## 快速开始

```bash
git clone <repo-url> vbt_project && cd vbt_project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_analytics.txt
# 树莓派额外安装:
pip install tflite-runtime
```

### 下载模型

```bash
mkdir -p models
# 从 TFHub 下载 MoveNet Lightning INT8:
# https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4
```

### 验证安装

```bash
.venv/bin/python3 run_tests.py
# 期望: 43 passed, 0 failed — ALL TESTS PASSED
```

### 实时监测

```bash
# USB 摄像头
.venv/bin/python3 vbt_realtime_main.py
# 树莓派 CSI 摄像头
.venv/bin/python3 vbt_main.py
```

### Web 看板

```bash
.venv/bin/streamlit run vbt_pro_coach_dashboard.py
```

---

## 训练模式

| 模式 | 速度区间 (m/s) | 停组阈值 | 推荐 Reps | 休息 |
|---|---|---|---|---|
| Power | 0.75–1.30 | 15% | 1–5 | 3 min |
| Strength | 0.15–0.50 | 20% | 2–6 | 4 min |
| Hypertrophy | 0.50–0.75 | 35% | 6–12 | 2 min |

看板侧边栏或 `vbt_training_modes.get_mode_policy()` 可切换模式。

---

## Quality Gate（数据信任层）

每个 rep 在写入分析前通过质量检查：

| 检查 | 拒绝原因码 |
|---|---|
| 标定使用兜底/超时方法 | `calibration_fallback` |
| 速度低于 MVT 地板 | `velocity_too_low` |
| ROM 完成度 < 60% | `rom_incomplete` |
| 杠铃路径偏移 > 25 cm | `unstable_tracking` |

`calib_is_fallback=1` 的 rep 在建模时默认排除。

---

## AI / DL 模块

### SetFatigueNet
- 架构: Conv1D(k=3,16) → GAP → Dense(8) → Dense(2, sigmoid)
- 输入: 组内每 rep 的 8 维特征向量（速度、ROM、速度损失、姿态分等）
- 输出: `fatigue_risk ∈ [0,1]`, `stop_probability ∈ [0,1]`, `confidence`
- 权重: `models/set_fatigue_net.npz`（纯 numpy，<100 KB）
- **无模型时**: 自动回退到速度衰减规则，记录 `model_version="rule_fallback"`

### TechniqueAnomalyNet
- 架构: Autoencoder (16→8→4→8→16)
- 输入: 16 维 rep 运动学特征向量
- 输出: `technique_anomaly_score`, `severity: normal/warning/high-risk`
- 权重: `models/technique_anomaly_net.npz`
- **无模型时**: 返回 score=0.0, severity="normal"

### 混合决策策略
- 安全规则优先（硬约束）
- DL 输出加权影响 increase/maintain/decrease/stop 决策
- 当 `dl_confidence < 0.3` 时回退到纯规则

---

## ML 训练流水线

```bash
# 1. 从数据库提取特征
.venv/bin/python3 vbt_ml_pipeline.py build-dataset --db squat_gym.db --out ml_data/

# 2. 训练 SetFatigueNet
.venv/bin/python3 vbt_ml_pipeline.py train-fatigue --data ml_data/ --out models/ --epochs 40

# 3. 训练 TechniqueAnomalyNet
.venv/bin/python3 vbt_ml_pipeline.py train-technique --data ml_data/ --out models/ --epochs 60

# 4. 评估指标
.venv/bin/python3 vbt_ml_pipeline.py evaluate --db squat_gym.db

# 5. 导出 metrics 报告
.venv/bin/python3 vbt_ml_pipeline.py report --db squat_gym.db --out ml_data/metrics.json
```

训练建议在桌面/笔记本上完成，将 `.npz` 权重文件复制到树莓派 `models/` 目录，Pi 上只做推理。

---

## DB Schema（新增表）

| 表 | 说明 |
|---|---|
| `sets` | 组级元数据：模式、负重、速度损失、推荐动作 |
| `ml_models` | 模型注册表：版本、路径、训练时间、指标 |
| `prediction_logs` | 每组 AI 预测日志：疲劳风险、停组概率、置信度 |
| `training_samples` | 特征 + 标签 + trust 元数据，供训练使用 |
| `user_feedback` | 用户 RPE 与主观感受标签 |

所有新表通过 `_ensure_ml_schema()` 幂等迁移，旧数据库零中断。

---

## 标定说明

| 方式 | 触发条件 | 精度 |
|---|---|---|
| 肩踝法（首选） | 站立时双肩 + 双踝可见 | 高 |
| 头踝法 | 鼻子 + 踝可见 | 中 |
| 肩髋法（兜底） | 踝被遮挡 | 低，标记 fallback |
| 超时兜底 | 50 帧内无法完成 | 极低 |

- `calib_method` + `calib_is_fallback` + `timing_source` 字段记录每 rep 的标定元数据
- 实时摄像头用 `time.monotonic()` 实测帧间隔；视频文件用 `1/video_fps`

---

## 测试

```bash
.venv/bin/python3 run_tests.py
```

覆盖范围（43 项）：
- letterbox 预处理 / CalibrationState 三路径 / SquatStateMachine
- sessions CRUD / migrate / schema 迁移
- Phase 2 回归（calib metadata、monotonic clock、depth bias）
- 训练模式策略（三模式有效性、未知模式回退）
- Quality Gate（6 种场景）
- SetLifecycleManager 生命周期
- 负重推荐（正常路径 + 安全覆盖）
- Session Report（聚合 + 空列表）
- DL 模型回退（SetFatigueNet + TechniqueAnomalyNet + InferenceWrapper）
- 混合策略安全覆盖
- DB ML 表结构（创建 + 列结构 + 迁移）

---

## .gitignore 约定

- `squat_gym.db`, `*.db-wal`, `*.db-shm` — 本地数据
- `videos/`, `recordings/` — 训练视频
- `models/*.tflite`, `models/*.npz` — 模型权重
- `ml_data/` — 训练数据集
- `.venv/` — 虚拟环境
