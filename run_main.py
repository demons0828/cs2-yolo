#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
游戏辅助程序启动脚本
支持GUI和命令行两种模式
"""

import sys
import os
import traceback

def check_dependencies():
    """检查依赖"""
    required_modules = [
        'numpy', 
        'cv2',
        'PIL',
        'pynput',
        'pyautogui',
        'onnxruntime'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("缺少以下依赖模块:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    """主函数"""
    try:
        # 检查基本依赖
        if not check_dependencies():
            return 1
        
        # 检查GUI依赖
        try:
            import tkinter
            has_gui = True
        except ImportError:
            has_gui = False
            print("警告: tkinter不可用，将使用命令行模式")
        
        # 根据参数和可用性决定启动模式
        if len(sys.argv) > 1 and sys.argv[1] == "--no-gui":
            # 强制命令行模式
            print("启动命令行模式...")
            from main import main as main_func
            main_func()
        elif has_gui:
            # GUI模式
            print("启动GUI模式...")
            try:
                from main_gui import main as gui_main
                gui_main()
            except Exception as e:
                print(f"GUI启动失败: {e}")
                print("切换到命令行模式...")
                from main import main as main_func
                main_func()
        else:
            # 命令行模式
            print("启动命令行模式...")
            from main import main as main_func
            main_func()
        
        return 0
        
    except Exception as e:
        print(f"启动失败: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())