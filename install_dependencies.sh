#!/bin/bash
# -*- coding: utf-8 -*-
"""
依赖安装脚本
用于安装游戏辅助程序所需的依赖
"""

echo "游戏辅助程序依赖安装脚本"
echo "=========================="

# 检查Python版本
echo "检查Python版本..."
python3 --version
if [ $? -ne 0 ]; then
    echo "错误: Python3未安装"
    exit 1
fi

# 检查pip
echo "检查pip..."
python3 -m pip --version
if [ $? -ne 0 ]; then
    echo "错误: pip未安装"
    exit 1
fi

# 创建虚拟环境
echo "创建虚拟环境..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "警告: 无法创建虚拟环境，尝试系统安装..."
    echo "请手动安装以下包:"
    echo "  sudo apt install python3-tk python3-pip"
    echo "  pip3 install -r requirements.txt --break-system-packages"
    exit 1
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "依赖安装成功！"
    echo ""
    echo "使用方法:"
    echo "1. 激活虚拟环境: source venv/bin/activate"
    echo "2. 启动程序: python main.py"
    echo "3. 或使用启动脚本: python run_main.py"
    echo ""
    echo "测试GUI:"
    echo "  python test_gui.py"
else
    echo "依赖安装失败！"
    echo "请检查网络连接和Python环境"
    exit 1
fi