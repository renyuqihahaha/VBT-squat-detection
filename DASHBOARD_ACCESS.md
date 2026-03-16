# VBT Coach Dashboard — MacBook 远程访问指南

## 在树莓派上启动看板

```bash
cd /home/kiki-pi/vbt_project
streamlit run vbt_pro_coach_dashboard.py
```

默认监听 `0.0.0.0:8501`，可从局域网内任意设备访问。

## 在 MacBook 上访问

1. **获取树莓派 IP**（在树莓派终端执行）：
   ```bash
   hostname -I | awk '{print $1}'
   ```
   例如：`192.168.1.100`

2. **在 MacBook 浏览器中打开**：
   ```
   http://<树莓派IP>:8501
   ```
   例如：`http://192.168.1.100:8501`

3. **确保同一局域网**：MacBook 与树莓派需在同一 WiFi 或局域网内。

## 可选：固定树莓派 IP

在路由器中为树莓派设置静态 IP，或使用 `hostname.local`：
```
http://kiki-pi.local:8501
```

## 生成 output_result.mp4

在「本地视频分析」模式下上传并处理深蹲视频后，分析结果会自动保存为 `output_result.mp4`，随后可在「视频+数据看板」中播放。
