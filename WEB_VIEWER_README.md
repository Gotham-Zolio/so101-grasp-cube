# Web Viewer 使用说明

## 概述

Web Viewer 允许你在无头模式（没有显示器或 GUI）下运行 ManiSkill 仿真，并通过浏览器查看实时画面。这对于在远程服务器或无显示环境中运行仿真非常有用。

## 功能特性

- 🌐 通过浏览器查看仿真画面
- 📸 截图功能
- 🎥 录制视频
- 📊 实时显示任务状态（模式、Episode、任务名称）
- 🚀 支持无头模式运行

## 快速开始

### 1. 运行 Web Viewer

```bash
uv run python hello_pick_cube_web.py
```

### 2. 在浏览器中查看

打开浏览器访问：
```
http://localhost:5000
```

### 3. 常用参数

```bash
# 运行 10 个 episodes，使用随机动作
uv run python hello_pick_cube_web.py --num_episodes 10 --random_actions

# 使用零动作
uv run python hello_pick_cube_web.py --zero_actions

# 使用 None 动作
uv run python hello_pick_cube_web.py --none_actions

# 修改端口
uv run python hello_pick_cube_web.py --port 8080

# 修改帧率
uv run python hello_pick_cube_web.py --fps 60

# 设置随机种子
uv run python hello_pick_cube_web.py --seed 42
```

## 在远程服务器上使用

### 方法 1: SSH 端口转发

如果你在远程服务器上运行，可以使用 SSH 端口转发：

```bash
# 在本地机器上
ssh -L 5000:localhost:5000 user@remote-server

# 然后在远程服务器上运行
uv run python hello_pick_cube_web.py

# 在本地浏览器访问
http://localhost:5000
```

### 方法 2: 直接访问

如果服务器有公网 IP 或在局域网内：

```bash
# 在服务器上运行
uv run python hello_pick_cube_web.py

# 在浏览器访问
http://server-ip:5000
```

## 网页功能

### 状态信息
- **Mode**: 当前模式（Initializing/Running/Finished）
- **Task**: 任务名称（PickCubeSO101）
- **Episode**: 当前 Episode / 总 Episode 数

### 控制按钮
- **Screenshot**: 保存当前所有相机视图的截图
- **Start/Stop Recording**: 开始/停止录制视频

## 文件输出

所有输出文件保存在：
- Linux/Mac: `~/tmp/outputs/web_viewer/`
- Windows: 用户的 temp 目录下

### 目录结构
```
~/tmp/outputs/web_viewer/
├── screenshots/
│   └── 2026-01-01_12-30-45/
│       └── render.jpg
└── videos/
    └── 2026-01-01_12-35-20/
        └── render.mp4
```

## 技术细节

### 架构
- Web Viewer 在后台线程中运行 HTTP 服务器
- 使用 MJPEG 流式传输实时画面
- 通过 REST API 进行状态更新和控制

### 性能
- 默认帧率: 30 FPS（可调）
- 流式传输帧率: 10 FPS（减少网络负载）
- 图像格式: JPEG（压缩传输）

### 兼容性
- ✅ 支持所有 ManiSkill 环境
- ✅ 支持 CPU 和 GPU 仿真
- ✅ 支持 Windows/Linux/Mac
- ✅ 无需显示器或 X11

## 集成到你的代码

你可以轻松地将 Web Viewer 集成到自己的代码中：

```python
from grasp_cube.utils.web_viewer import WebViewer
import gymnasium as gym

# 创建并启动 Web Viewer
viewer = WebViewer(port=5000)
viewer.start()

# 创建环境（使用 rgb_array 模式）
env = gym.make(
    "PickCubeSO101-v1",
    render_mode="rgb_array"
)

# 更新状态
viewer.update_status(
    mode="Running",
    episode=1,
    total_episodes=10,
    task="PickCubeSO101"
)

# 在循环中更新帧
obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 渲染并推送到 web viewer
    rgb_frame = env.render()
    if rgb_frame is not None:
        viewer.update_frames({"render": rgb_frame})
    
    if terminated or truncated:
        break
```

## 常见问题

### Q: 端口被占用 (Address already in use)
A: 这是最常见的问题。使用 `--port` 参数指定其他端口：
```bash
uv run python hello_pick_cube_web.py --port 5001 --num_episodes 2 --random_actions
```

或者找到并关闭占用端口的进程：
```bash
# 查找占用端口 5000 的进程
lsof -i :5000
# 或
netstat -tulpn | grep :5000

# 关闭进程（替换 <PID> 为实际进程 ID）
kill <PID>
```

### Q: 浏览器显示"No Signal"
A: 确保环境已经开始渲染，并且使用了 `render_mode="rgb_array"`

### Q: 画面卡顿
A: 可以尝试降低 `--fps` 参数，或者检查网络连接

### Q: 在 Windows 上无法访问
A: 确保防火墙允许该端口，或使用 `http://localhost:5000` 而不是 `http://0.0.0.0:5000`

## 与原始 hello_pick_cube.py 的区别

| 特性 | hello_pick_cube.py | hello_pick_cube_web.py |
|------|-------------------|----------------------|
| 渲染模式 | `human` (需要显示器) | `rgb_array` (无头模式) |
| 查看方式 | SAPIEN Viewer 窗口 | 浏览器 Web 界面 |
| 远程访问 | ❌ 不支持 | ✅ 支持 |
| 截图/录制 | 手动 | 一键操作 |
| 状态监控 | 终端输出 | Web 界面 |
| Episode 控制 | 手动中断 | 自动运行多个 episodes |

## 示例：训练可视化

Web Viewer 也可以用于可视化训练过程：

```python
from grasp_cube.utils.web_viewer import WebViewer

viewer = WebViewer(port=5000)
viewer.start()

for episode in range(num_episodes):
    viewer.update_status(
        mode="Training",
        episode=episode + 1,
        total_episodes=num_episodes,
        task="PickCubeSO101"
    )
    
    # 训练代码...
    # 定期更新画面
    if episode % render_interval == 0:
        rgb_frame = env.render()
        viewer.update_frames({"render": rgb_frame})
```

## 相关文件

- `grasp_cube/utils/web_viewer/viewer.py` - Web Viewer 核心实现
- `hello_pick_cube_web.py` - 使用示例
- `hello_pick_cube.py` - 原始版本（需要显示器）
