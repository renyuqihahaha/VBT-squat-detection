#!/usr/bin/env bash
set -Eeuo pipefail

# VBT-squat-detection Raspberry Pi 一键部署脚本
# 用法：
#   bash deploy_raspberrypi.sh usb   # USB 摄像头，运行 vbt_realtime_main.py
#   bash deploy_raspberrypi.sh csi   # CSI 摄像头，运行 vbt_main.py
# 可选环境变量：
#   REPO_DIR=$HOME/VBT-squat-detection
#   BRANCH=feature/raspberrypi_realtime
#   REPO_URL=https://github.com/renyuqihahaha/VBT-squat-detection.git

MODE="${1:-usb}"
REPO_DIR="${REPO_DIR:-$HOME/VBT-squat-detection}"
BRANCH="${BRANCH:-feature/raspberrypi_realtime}"
REPO_URL="${REPO_URL:-https://github.com/renyuqihahaha/VBT-squat-detection.git}"
LOG_FILE="${LOG_FILE:-realtime.log}"

case "$MODE" in
  usb) ENTRYPOINT="vbt_realtime_main.py" ;;
  csi) ENTRYPOINT="vbt_main.py" ;;
  *) echo "用法：$0 [usb|csi]" >&2; exit 2 ;;
esac

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

log "摄像头检测："
if command -v libcamera-hello >/dev/null 2>&1; then
  log "CSI 测试命令可用：libcamera-hello --timeout 2000"
else
  log "未发现 libcamera-hello；如使用 CSI，请先安装 Raspberry Pi camera stack"
fi
if command -v v4l2-ctl >/dev/null 2>&1; then
  v4l2-ctl --list-devices || true
else
  log "未发现 v4l2-ctl；USB 摄像头可用 OpenCV 程序进一步测试"
fi

if [ ! -d "$REPO_DIR/.git" ]; then
  log "克隆仓库到 $REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
log "切换并更新分支：$BRANCH"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

log "准备 Python 虚拟环境"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements_analytics.txt
python -m pip install numpy opencv-python matplotlib streamlit pandas
python -m pip install tflite-runtime picamera2 || {
  log "pip 安装 tflite-runtime/picamera2 失败；尝试使用系统包（需要 sudo 权限）"
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3-picamera2 python3-libcamera || true
  fi
}

if [ ! -f vbt_config.json ]; then
  if [ -f vbt_config.example.json ]; then
    cp vbt_config.example.json vbt_config.json
  else
    cat > vbt_config.json <<'JSON'
{
  "active_user_name": "pi_user",
  "current_load_kg": 60.0,
  "user_height_cm": 175.0
}
JSON
  fi
  log "已生成本地私有配置 vbt_config.json，请按实际用户/负重/身高修改"
fi

# 私有配置只在本地保留：即使仓库曾跟踪该文件，也避免本机改动被提交。
git update-index --skip-worktree vbt_config.json 2>/dev/null || true

log "停止旧的实时采集进程"
pkill -f 'vbt_realtime_main.py' 2>/dev/null || true
pkill -f 'vbt_main.py' 2>/dev/null || true
sleep 1

log "后台启动 $ENTRYPOINT，日志：$REPO_DIR/$LOG_FILE"
nohup .venv/bin/python3 "$ENTRYPOINT" > "$LOG_FILE" 2>&1 &
echo $! > realtime.pid

log "启动完成：PID=$(cat realtime.pid)"
log "远程查看日志：ssh pi@<树莓派IP> 'tail -f $REPO_DIR/$LOG_FILE'"
