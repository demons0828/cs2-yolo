#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
游戏辅助程序GUI启动脚本
"""

import sys
import os
import traceback

def main():
    """主函数"""
    try:
        # 检查依赖
        required_modules = [
            'tkinter',
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
            return 1
        
        # 导入并启动GUI
        from main_gui import main as gui_main
        gui_main()
        return 0
        
    except Exception as e:
        print(f"启动失败: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())