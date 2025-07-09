# 游戏辅助程序GUI界面

## 概述

游戏辅助程序现在提供了图形用户界面(GUI)，让用户可以更方便地配置参数、启动/停止检测，并实时查看运行状态。

## 快速开始

### 1. 安装依赖
```bash
./install_dependencies.sh
```

### 2. 启动程序
```bash
./quick_start.sh
```

或者手动启动：
```bash
source venv/bin/activate
python main.py
```

## 功能特性

### 控制面板
- **模型配置**: 选择ONNX模型文件路径
- **检测参数**: 调整置信度阈值和NMS阈值
- **屏幕配置**: 设置输入尺寸和屏幕裁剪尺寸
- **操作按钮**: 启动/停止检测、保存/重新加载配置

### 状态显示
- **实时状态**: 显示程序运行状态
- **FPS显示**: 显示检测帧率
- **目标计数**: 显示检测到的目标数量
- **运行日志**: 实时显示程序运行日志

### 快捷键说明
- **E**: 瞄准最近目标
- **Q**: 退出程序
- **O**: 切换T/CT模式
- **I**: 切换图像显示

## 启动方式

### 方式1: 快速启动（推荐）
```bash
./quick_start.sh
```
自动检测环境并启动合适的模式。

### 方式2: 直接启动GUI
```bash
source venv/bin/activate
python main.py
```
程序会自动启动GUI界面。

### 方式3: 使用专用启动脚本
```bash
source venv/bin/activate
python run_main.py
```

### 方式4: 命令行模式
```bash
source venv/bin/activate
python main.py --no-gui
```

### 方式5: 测试GUI
```bash
source venv/bin/activate
python test_gui.py
```

## 安装说明

### 自动安装
```bash
./install_dependencies.sh
```

### 手动安装
1. 创建虚拟环境：
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 安装GUI支持（可选）：
   ```bash
   sudo apt install python3-tk
   ```

## 使用说明

### 1. 配置模型
1. 点击"浏览"按钮选择ONNX模型文件
2. 或直接在文本框中输入模型路径

### 2. 调整参数
- **置信度阈值**: 控制检测的敏感度（0.1-1.0）
- **NMS阈值**: 控制重叠框的抑制程度（0.1-1.0）
- **输入尺寸**: 模型输入图像的尺寸
- **裁剪尺寸**: 屏幕捕获区域的尺寸

### 3. 启动检测
1. 确保配置正确
2. 点击"启动检测"按钮
3. 程序会在后台运行检测
4. 使用快捷键E进行瞄准

### 4. 监控状态
- 查看状态栏了解程序运行状态
- 观察日志窗口获取详细信息
- 可以保存日志文件用于调试

### 5. 停止检测
- 点击"停止检测"按钮
- 或按Q键退出程序

## 配置管理

### 保存配置
点击"保存配置"按钮将当前设置保存到`config.json`文件。

### 重新加载配置
点击"重新加载配置"按钮从`config.json`文件加载设置。

## 日志功能

### 查看日志
程序运行时的所有信息都会显示在日志窗口中。

### 清空日志
点击"清空日志"按钮清除日志显示。

### 保存日志
点击"保存日志"按钮将日志保存到文件。

## 故障排除

### GUI无法启动
1. 检查是否安装了所有依赖：
   ```bash
   ./install_dependencies.sh
   ```
2. 确保系统支持tkinter
3. 尝试使用命令行模式：`python main.py --no-gui`

### 检测无法启动
1. 检查模型文件路径是否正确
2. 确认模型文件存在且可访问
3. 查看日志窗口的错误信息

### 性能问题
1. 调整置信度阈值和NMS阈值
2. 减小输入尺寸或裁剪尺寸
3. 检查系统资源使用情况

## 文件结构

```
├── main.py                    # 主程序文件
├── main_gui.py               # GUI界面文件
├── run_main.py               # 启动脚本
├── test_gui.py               # GUI测试程序
├── install_dependencies.sh   # 依赖安装脚本
├── quick_start.sh            # 快速启动脚本
├── config.json               # 配置文件
├── requirements.txt          # 依赖列表
└── README_GUI_MAIN.md        # 本文档
```

## 依赖要求

- Python 3.7+
- tkinter (通常随Python安装)
- numpy
- opencv-python
- Pillow
- pynput
- pyautogui
- onnxruntime

## 注意事项

1. 首次运行时会自动创建默认配置文件
2. 程序运行时会在后台进行屏幕捕获
3. 使用快捷键E进行瞄准时，确保目标在屏幕中心区域
4. 建议在游戏窗口模式下使用以获得最佳效果
5. 程序需要适当的权限来访问屏幕和鼠标

## 更新日志

- v1.0: 初始GUI版本
  - 添加图形界面
  - 支持参数配置
  - 实时状态显示
  - 日志功能
  - 自动安装脚本
  - 快速启动脚本