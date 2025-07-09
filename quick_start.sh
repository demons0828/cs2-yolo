#!/bin/bash
# -*- coding: utf-8 -*-
"""
快速启动脚本
自动检测环境并启动游戏辅助程序
"""

echo "游戏辅助程序快速启动"
echo "===================="

# 检查虚拟环境是否存在
if [ -d "venv" ]; then
    echo "发现虚拟环境，激活中..."
    source venv/bin/activate
    echo "虚拟环境已激活"
else
    echo "未发现虚拟环境"
    echo "请先运行安装脚本: ./install_dependencies.sh"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import numpy, cv2, PIL, pynput, pyautogui, onnxruntime; print('依赖检查通过')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "依赖检查失败，请先安装依赖:"
    echo "  ./install_dependencies.sh"
    exit 1
fi

# 检查GUI支持
echo "检查GUI支持..."
python3 -c "import tkinter; print('GUI支持正常')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "启动GUI模式..."
    python3 main.py
else
    echo "GUI不可用，启动命令行模式..."
    python3 main.py --no-gui
fi