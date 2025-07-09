#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的GUI测试程序
用于验证GUI功能是否正常工作
"""

import sys
import os

def test_tkinter():
    """测试tkinter是否可用"""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        
        # 创建主窗口
        root = tk.Tk()
        root.title("GUI测试")
        root.geometry("400x300")
        
        # 创建标签
        label = ttk.Label(root, text="GUI测试成功！\n\n游戏辅助程序GUI功能正常。", 
                         font=('Arial', 12), justify='center')
        label.pack(expand=True)
        
        # 创建按钮
        def show_info():
            messagebox.showinfo("信息", "GUI功能正常！")
        
        button = ttk.Button(root, text="测试按钮", command=show_info)
        button.pack(pady=20)
        
        # 创建退出按钮
        exit_button = ttk.Button(root, text="退出", command=root.destroy)
        exit_button.pack(pady=10)
        
        print("GUI测试程序启动成功！")
        root.mainloop()
        return True
        
    except ImportError as e:
        print(f"tkinter不可用: {e}")
        return False
    except Exception as e:
        print(f"GUI测试失败: {e}")
        return False

def main():
    """主函数"""
    print("开始GUI测试...")
    
    if test_tkinter():
        print("GUI测试成功！")
        return 0
    else:
        print("GUI测试失败！")
        return 1

if __name__ == "__main__":
    sys.exit(main())