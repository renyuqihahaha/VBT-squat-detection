# VBT-squat-detection

基于速度的深蹲监测系统（Velocity Based Training, VBT），运行在树莓派上的实时视觉算法，通过测量深蹲向心阶段的平均速度，帮助你更科学地控制训练强度与疲劳水平。

---

## 技术栈

- **语言与运行环境**：Python 3（推荐在树莓派上使用虚拟环境 `.venv`）
- **计算机视觉**：OpenCV
- **姿态估计 / 关键点检测**：MoveNet (TensorFlow Lite)
- **数据存储**：SQLite (`squat_gym.db`)
- **可视化**：Matplotlib

> 说明：项目最初参考了 VBT 理论与 MediaPipe 等方案，但当前实现基于 MoveNet + 手写计数与分析逻辑。

---

## 目录结构与模块说明

- **核心逻辑（实时与批处理）**
  - `vbt_main.py`：
    - 在树莓派上通过 Picamera2 实时采集画面，运行 MoveNet 关键点检测。
    - 估计髋关节纵向位移，实时计数深蹲 rep，并显示当前平均向心速度 (MCV)。
  - `vbt_analytics_pro.py`：
    - 增强版分析内核：膝关节角度、躯干倾角、DTW 相似度、SQLite 持久化与 Matplotlib 曲线报表。
    - 定义数据库表结构（`reps` / `standard_action`），实现标准动作对比等功能。
  - `vbt_video_processor.py`：
    - 离线视频批处理器：扫描 `videos/` 目录，对每个视频自动计数深蹲 reps。
    - 计算每个 rep 的 MCV、ROM、最小膝角、躯干最大倾角和 DTW 相似度，并写入 `batch_reps` 表。

- **教练仪表盘 / 训练后分析**
  - `vbt_fatigue_analyst.py`：
    - 从 `batch_reps` 或 `reps` 读取最近一次训练的速度序列，计算速度衰减率。
    - 生成 `velocity_loss_report.png`，并给出是否应该停止本组训练的建议。
  - `vbt_pro_coach_dashboard.py`：
    - 汇总整个 `squat_gym.db` 的训练数据，按 session 统计：总 reps、最佳速度、平均速度、速度损失%、动作一致性等。
    - 生成综合报表 `vbt_pro_coach_report.png`，并在终端打印 Markdown 形式的教练视图和下次训练建议。
  - `streamlit_dashboard.py`：
    - 基于同一数据逻辑的 **Web 可视化看板**：读取 `squat_gym.db`，展示 MCV 折线图、总次数/平均速度/速度损失率，并可通过下拉菜单切换历史 Session。

- **数据维护与标准动作工具**
  - `set_standard_action.py`：从指定视频中提取完整动作序列，写入 `standard_action` 表作为“标准深蹲曲线”。
  - `clean_batch_reps.py`：清洗 `batch_reps` 表中膝角过大（未真正下蹲）的误判记录。
  - `merge_batch_reps_by_session.py`：按规范化的 session 名合并同一 session 的多条 `filename` 记录，并重排 rep 编号。

- **模型与数据（默认不纳入 Git）**
  - `models/movenet_lightning.tflite`：MoveNet TensorFlow Lite 模型文件。
  - `squat_gym.db`：SQLite 数据库（已在 `.gitignore` 中忽略，本地运行时自动生成 / 更新）。
  - `videos/`：训练视频素材目录（也在 `.gitignore` 中忽略）。

---

## 如何开始（在树莓派上运行 `vbt_main.py`）

1. **克隆项目并进入目录**

   ```bash
   git clone <your-repo-url> vbt_project
   cd vbt_project
   ```

2. **创建并激活虚拟环境（推荐）**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **安装依赖**

   使用项目提供的依赖文件：

   ```bash
   pip install -r requirements_analytics.txt
   ```

   > 若在树莓派上运行摄像头模式，请确保额外安装 `tflite_runtime` 与 `picamera2`，并使用 `numpy<2` 的版本。

4. **准备模型与摄像头**

   - 确保 `models/movenet_lightning.tflite` 已存在（若无，可从官方 MoveNet TFLite 模型下载）。
   - 树莓派连接好 CSI/USB 摄像头，Picamera2 可正常工作。

5. **运行实时深蹲监测**

   ```bash
   python3 vbt_main.py
   ```

   或在虚拟环境中：

   ```bash
   .venv/bin/python vbt_main.py
   ```

   启动后：

   - 屏幕会显示实时画面与关键指标（rep 计数、当前平均向心速度等）。
   - 每个 rep 的数据会记录到本地 SQLite 数据库 `squat_gym.db` 中，供后续疲劳分析与教练仪表盘使用。

---

如需离线分析历史视频或生成教练仪表盘，可分别运行：

```bash
# 批量处理 videos/ 目录下的视频
.venv/bin/python vbt_video_processor.py

# 生成疲劳分析报告
.venv/bin/python vbt_fatigue_analyst.py

# 生成教练视图与综合报表
.venv/bin/python vbt_pro_coach_dashboard.py

# 启动 Web 可视化看板（需安装 streamlit）
.venv/bin/streamlit run streamlit_dashboard.py
```

