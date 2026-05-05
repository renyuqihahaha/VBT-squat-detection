#!/usr/bin/env bash
set -Eeuo pipefail

# VBT-squat-detection 自动更新 + 依赖安装 + 重启实时采集
# 用法：
#   ./update_and_run.sh usb   # USB 摄像头，运行 vbt_realtime_main.py
#   ./update_and_run.sh csi   # CSI 摄像头，运行 vbt_main.py

MODE="${1:-usb}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
BRANCH="${BRANCH:-feature/raspberrypi_realtime}"
LOG_FILE="${LOG_FILE:-realtime.log}"

case "$MODE" in
  usb) ENTRYPOINT="vbt_realtime_main.py" ;;
  csi) ENTRYPOINT="vbt_main.py" ;;
  *) echo "用法：$0 [usb|csi]" >&2; exit 2 ;;
esac

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

cd "$REPO_DIR"
log "拉取 GitHub 分支：$BRANCH"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

if [ ! -d .venv ]; then
  log "创建虚拟环境 .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements_analytics.txt
python -m pip install tflite-runtime picamera2 || {
  log "pip 安装树莓派摄像头依赖失败；如使用 CSI，请确认 apt 包 python3-picamera2/python3-libcamera 已安装"
}

if [ ! -f vbt_config.json ]; then
  cp vbt_config.example.json vbt_config.json
  log "已由模板生成 vbt_config.json，请按本地用户配置修改"
fi
git update-index --skip-worktree vbt_config.json 2>/dev/null || true

log "停止旧进程"
pkill -f 'vbt_realtime_main.py' 2>/dev/null || true
pkill -f 'vbt_main.py' 2>/dev/null || true
sleep 1

log "启动 $ENTRYPOINT，日志输出到 $LOG_FILE"
nohup .venv/bin/python3 "$ENTRYPOINT" > "$LOG_FILE" 2>&1 &
echo $! > realtime.pid
log "启动完成：PID=$(cat realtime.pid)"
log "查看日志：tail -f $LOG_FILE"
