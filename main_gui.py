#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
游戏辅助程序GUI界面
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import threading
import time
import logging
from typing import Dict, Any, Optional
import os
import sys

# 导入main.py中的类
from main import GameAssistant, Config

class GameAssistantGUI:
    """游戏辅助程序GUI界面"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("游戏辅助程序")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # 初始化组件
        self.config = Config()
        self.assistant = None
        self.is_running = False
        self.assistant_thread = None
        
        # 设置样式
        self.setup_styles()
        
        # 创建界面
        self.create_widgets()
        
        # 绑定事件
        self.bind_events()
        
        # 启动状态更新
        self.update_status()
        
    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Info.TLabel', font=('Arial', 10))
        style.configure('Status.TLabel', font=('Arial', 9))
        
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="游戏辅助程序", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 左侧控制面板
        self.create_control_panel(main_frame)
        
        # 右侧显示面板
        self.create_display_panel(main_frame)
        
        # 底部状态栏
        self.create_status_bar(main_frame)
    
    def create_control_panel(self, parent):
        """创建左侧控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        
        # 模型配置
        model_frame = ttk.LabelFrame(control_frame, text="模型配置", padding="5")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="模型路径:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.model_path_var = tk.StringVar(value=self.config.config.get("model_path", ""))
        ttk.Entry(model_frame, textvariable=self.model_path_var, width=30).grid(row=0, column=1, padx=(5, 0), pady=2)
        ttk.Button(model_frame, text="浏览", command=self.browse_model).grid(row=0, column=2, padx=(5, 0), pady=2)
        
        # 检测参数
        params_frame = ttk.LabelFrame(control_frame, text="检测参数", padding="5")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(params_frame, text="置信度阈值:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.conf_threshold_var = tk.DoubleVar(value=self.config.config.get("conf_threshold", 0.4))
        conf_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, variable=self.conf_threshold_var, orient=tk.HORIZONTAL)
        conf_scale.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=2)
        self.conf_label = ttk.Label(params_frame, text=f"{self.conf_threshold_var.get():.2f}")
        self.conf_label.grid(row=0, column=2, padx=(5, 0), pady=2)
        
        ttk.Label(params_frame, text="NMS阈值:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.nms_threshold_var = tk.DoubleVar(value=self.config.config.get("nms_threshold", 0.4))
        nms_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, variable=self.nms_threshold_var, orient=tk.HORIZONTAL)
        nms_scale.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=2)
        self.nms_label = ttk.Label(params_frame, text=f"{self.nms_threshold_var.get():.2f}")
        self.nms_label.grid(row=1, column=2, padx=(5, 0), pady=2)
        
        # 屏幕配置
        screen_frame = ttk.LabelFrame(control_frame, text="屏幕配置", padding="5")
        screen_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(screen_frame, text="输入尺寸:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.input_size_x_var = tk.IntVar(value=self.config.config.get("input_size", [640, 640])[0])
        self.input_size_y_var = tk.IntVar(value=self.config.config.get("input_size", [640, 640])[1])
        ttk.Entry(screen_frame, textvariable=self.input_size_x_var, width=8).grid(row=0, column=1, padx=(5, 2), pady=2)
        ttk.Label(screen_frame, text="x").grid(row=0, column=2, pady=2)
        ttk.Entry(screen_frame, textvariable=self.input_size_y_var, width=8).grid(row=0, column=3, padx=(2, 0), pady=2)
        
        ttk.Label(screen_frame, text="裁剪尺寸:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.crop_size_x_var = tk.IntVar(value=self.config.config.get("screen_crop_size", [1920, 1080])[0])
        self.crop_size_y_var = tk.IntVar(value=self.config.config.get("screen_crop_size", [1920, 1080])[1])
        ttk.Entry(screen_frame, textvariable=self.crop_size_x_var, width=8).grid(row=1, column=1, padx=(5, 2), pady=2)
        ttk.Label(screen_frame, text="x").grid(row=1, column=2, pady=2)
        ttk.Entry(screen_frame, textvariable=self.crop_size_y_var, width=8).grid(row=1, column=3, padx=(2, 0), pady=2)
        
        # 配置网格权重
        params_frame.columnconfigure(1, weight=1)
        screen_frame.columnconfigure(1, weight=1)
        screen_frame.columnconfigure(3, weight=1)
        
        # 操作按钮
        action_frame = ttk.LabelFrame(control_frame, text="操作", padding="5")
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_btn = ttk.Button(action_frame, text="启动检测", command=self.start_detection)
        self.start_btn.pack(fill=tk.X, pady=2)
        
        self.stop_btn = ttk.Button(action_frame, text="停止检测", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="保存配置", command=self.save_config).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="重新加载配置", command=self.reload_config).pack(fill=tk.X, pady=2)
        
        # 快捷键说明
        hotkey_frame = ttk.LabelFrame(control_frame, text="快捷键说明", padding="5")
        hotkey_frame.pack(fill=tk.X, pady=(0, 10))
        
        hotkeys = [
            ("E", "瞄准最近目标"),
            ("Q", "退出程序"),
            ("O", "切换T/CT模式"),
            ("I", "切换图像显示")
        ]
        
        for i, (key, desc) in enumerate(hotkeys):
            ttk.Label(hotkey_frame, text=f"{key}:", font=('Arial', 9, 'bold')).grid(row=i, column=0, sticky=tk.W, pady=1)
            ttk.Label(hotkey_frame, text=desc, font=('Arial', 9)).grid(row=i, column=1, sticky=tk.W, padx=(10, 0), pady=1)
    
    def create_display_panel(self, parent):
        """创建右侧显示面板"""
        display_frame = ttk.LabelFrame(parent, text="状态显示", padding="10")
        display_frame.grid(row=1, column=1, sticky="nsew")
        
        # 状态信息
        status_frame = ttk.Frame(display_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="状态: 未启动", style='Status.TLabel')
        self.status_label.pack(anchor=tk.W)
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0.0", style='Status.TLabel')
        self.fps_label.pack(anchor=tk.W)
        
        self.detection_label = ttk.Label(status_frame, text="检测目标: 0", style='Status.TLabel')
        self.detection_label.pack(anchor=tk.W)
        
        # 日志显示
        log_frame = ttk.LabelFrame(display_frame, text="运行日志", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, width=60)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 日志按钮
        log_btn_frame = ttk.Frame(log_frame)
        log_btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(log_btn_frame, text="清空日志", command=self.clear_log).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(log_btn_frame, text="保存日志", command=self.save_log).pack(side=tk.LEFT)
    
    def create_status_bar(self, parent):
        """创建底部状态栏"""
        status_bar = ttk.Frame(parent)
        status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        self.status_bar_label = ttk.Label(status_bar, text="就绪", style='Status.TLabel')
        self.status_bar_label.pack(side=tk.LEFT)
        
        # 配置网格权重
        status_bar.columnconfigure(0, weight=1)
    
    def bind_events(self):
        """绑定事件"""
        # 绑定配置变更事件
        self.conf_threshold_var.trace('w', self.on_conf_change)
        self.nms_threshold_var.trace('w', self.on_nms_change)
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_conf_change(self, *args):
        """置信度阈值变更"""
        self.conf_label.config(text=f"{self.conf_threshold_var.get():.2f}")
    
    def on_nms_change(self, *args):
        """NMS阈值变更"""
        self.nms_label.config(text=f"{self.nms_threshold_var.get():.2f}")
    
    def browse_model(self):
        """浏览模型文件"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("ONNX files", "*.onnx"), ("All files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def start_detection(self):
        """启动检测"""
        try:
            if self.is_running:
                return
            
            # 更新配置
            self.update_config_from_gui()
            
            # 创建游戏辅助实例
            self.assistant = GameAssistant()
            
            # 在新线程中运行检测
            self.assistant_thread = threading.Thread(target=self.run_detection, daemon=True)
            self.assistant_thread.start()
            
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            self.log_message("检测已启动")
            self.status_label.config(text="状态: 运行中")
            
        except Exception as e:
            messagebox.showerror("错误", f"启动检测失败: {e}")
            self.log_message(f"启动失败: {e}")
    
    def run_detection(self):
        """运行检测（在独立线程中）"""
        try:
            if self.assistant:
                self.assistant.run()
        except Exception as e:
            self.log_message(f"检测运行错误: {e}")
        finally:
            # 在主线程中更新UI
            self.root.after(0, self.detection_finished)
    
    def detection_finished(self):
        """检测完成回调"""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="状态: 已停止")
        self.log_message("检测已停止")
    
    def stop_detection(self):
        """停止检测"""
        try:
            if self.assistant:
                self.assistant.running = False
                self.log_message("正在停止检测...")
        except Exception as e:
            messagebox.showerror("错误", f"停止检测失败: {e}")
            self.log_message(f"停止失败: {e}")
    
    def update_config_from_gui(self):
        """从GUI更新配置"""
        try:
            self.config.config.update({
                "model_path": self.model_path_var.get(),
                "conf_threshold": self.conf_threshold_var.get(),
                "nms_threshold": self.nms_threshold_var.get(),
                "input_size": [self.input_size_x_var.get(), self.input_size_y_var.get()],
                "screen_crop_size": [self.crop_size_x_var.get(), self.crop_size_y_var.get()]
            })
        except Exception as e:
            self.log_message(f"更新配置失败: {e}")
    
    def save_config(self):
        """保存配置"""
        try:
            self.update_config_from_gui()
            self.config.save_config(self.config.config)
            messagebox.showinfo("成功", "配置已保存")
            self.log_message("配置已保存")
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {e}")
            self.log_message(f"保存配置失败: {e}")
    
    def reload_config(self):
        """重新加载配置"""
        try:
            self.config.config = self.config.load_config()
            
            # 更新GUI显示
            self.model_path_var.set(self.config.config.get("model_path", ""))
            self.conf_threshold_var.set(self.config.config.get("conf_threshold", 0.4))
            self.nms_threshold_var.set(self.config.config.get("nms_threshold", 0.4))
            
            input_size = self.config.config.get("input_size", [640, 640])
            self.input_size_x_var.set(input_size[0])
            self.input_size_y_var.set(input_size[1])
            
            crop_size = self.config.config.get("screen_crop_size", [1920, 1080])
            self.crop_size_x_var.set(crop_size[0])
            self.crop_size_y_var.set(crop_size[1])
            
            messagebox.showinfo("成功", "配置已重新加载")
            self.log_message("配置已重新加载")
        except Exception as e:
            messagebox.showerror("错误", f"重新加载配置失败: {e}")
            self.log_message(f"重新加载配置失败: {e}")
    
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
    
    def save_log(self):
        """保存日志"""
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            title="保存日志",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("成功", "日志已保存")
            except Exception as e:
                messagebox.showerror("错误", f"保存日志失败: {e}")
    
    def log_message(self, message: str):
        """添加日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # 在主线程中更新UI
        self.root.after(0, lambda: self.log_text.insert(tk.END, log_entry))
        self.root.after(0, lambda: self.log_text.see(tk.END))
    
    def update_status(self):
        """更新状态显示"""
        try:
            if self.is_running and self.assistant:
                # 这里可以添加更多状态信息的更新
                pass
        except Exception as e:
            pass
        
        # 每秒更新一次状态
        self.root.after(1000, self.update_status)
    
    def on_closing(self):
        """窗口关闭事件"""
        try:
            if self.is_running:
                if messagebox.askokcancel("确认", "检测正在运行，确定要退出吗？"):
                    self.stop_detection()
                    self.root.after(1000, self.root.destroy)
            else:
                self.root.destroy()
        except Exception as e:
            self.root.destroy()

def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = GameAssistantGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"GUI启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()