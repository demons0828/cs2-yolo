import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import threading
import time
import os
import sys
from typing import Dict, Any, Optional, List
import logging
from device_manager import get_device_manager
from performance_monitor import get_performance_monitor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGameAssistantGUI:
    """增强的游戏辅助程序GUI"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("游戏辅助程序 - 增强版")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # 初始化组件
        self.device_manager = get_device_manager()
        self.performance_monitor = get_performance_monitor()
        self.config = self.load_config()
        self.assistant_thread = None
        self.is_running = False
        
        # 设置样式
        self.setup_styles()
        
        # 创建界面
        self.create_widgets()
        
        # 刷新设备信息
        self.refresh_device_info()
        
        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#7f8c8d')
        style.configure('Success.TLabel', font=('Arial', 10), foreground='#27ae60')
        style.configure('Warning.TLabel', font=('Arial', 10), foreground='#f39c12')
        style.configure('Error.TLabel', font=('Arial', 10), foreground='#e74c3c')
        
        # 按钮样式
        style.configure('Primary.TButton', font=('Arial', 10, 'bold'))
        style.configure('Success.TButton', background='#27ae60')
        style.configure('Danger.TButton', background='#e74c3c')
        
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self.get_default_config()
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "model_path": "onnxmd/10w320v5.onnx",
            "conf_threshold": 0.4,
            "nms_threshold": 0.4,
            "input_size": [640, 640],
            "screen_crop_size": [1920, 1080],
            "providers": [self.device_manager.recommended_provider],
            "provider_options": [self.device_manager.get_provider_config(self.device_manager.recommended_provider)],
            "names": {"0": "t_body", "1": "t_head", "2": "ct_body", "3": "ct_head"},
            "inference_mode": "auto",
            "auto_optimize": True,
            "performance_monitoring": True
        }
    
    def save_config(self):
        """保存配置"""
        try:
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            logger.info("配置已保存")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            messagebox.showerror("错误", f"保存配置失败: {e}")
    
    def create_widgets(self):
        """创建界面组件"""
        # 创建主容器
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧面板
        left_frame = ttk.Frame(main_container)
        main_container.add(left_frame, weight=1)
        
        # 右侧面板
        right_frame = ttk.Frame(main_container)
        main_container.add(right_frame, weight=1)
        
        # 创建左侧内容
        self.create_left_panel(left_frame)
        
        # 创建右侧内容
        self.create_right_panel(right_frame)
    
    def create_left_panel(self, parent):
        """创建左侧面板"""
        # 配置部分
        config_frame = ttk.LabelFrame(parent, text="配置设置", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_model_config(config_frame)
        self.create_inference_config(config_frame)
        self.create_detection_config(config_frame)
        
        # 设备信息部分
        device_frame = ttk.LabelFrame(parent, text="设备信息", padding="10")
        device_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.create_device_info(device_frame)
        
        # 控制按钮部分
        control_frame = ttk.LabelFrame(parent, text="程序控制", padding="10")
        control_frame.pack(fill=tk.X)
        
        self.create_control_buttons(control_frame)
    
    def create_model_config(self, parent):
        """创建模型配置"""
        model_frame = ttk.Frame(parent)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="模型配置", style='Subtitle.TLabel').pack(anchor=tk.W)
        
        # 模型路径
        path_frame = ttk.Frame(model_frame)
        path_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(path_frame, text="模型路径:").pack(side=tk.LEFT)
        self.model_path_var = tk.StringVar(value=self.config.get("model_path", ""))
        model_path_entry = ttk.Entry(path_frame, textvariable=self.model_path_var, width=40)
        model_path_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        
        ttk.Button(path_frame, text="浏览", command=self.browse_model_file).pack(side=tk.RIGHT)
        
        # 模型列表
        models_frame = ttk.Frame(model_frame)
        models_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(models_frame, text="可用模型:").pack(side=tk.LEFT)
        self.model_list_var = tk.StringVar()
        model_combo = ttk.Combobox(models_frame, textvariable=self.model_list_var, state="readonly")
        model_combo.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        
        # 刷新模型列表
        self.refresh_model_list(model_combo)
        model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)
    
    def create_inference_config(self, parent):
        """创建推理配置"""
        inference_frame = ttk.Frame(parent)
        inference_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(inference_frame, text="推理配置", style='Subtitle.TLabel').pack(anchor=tk.W)
        
        # 推理模式
        mode_frame = ttk.Frame(inference_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mode_frame, text="推理模式:").pack(side=tk.LEFT)
        self.inference_mode_var = tk.StringVar(value=self.config.get("inference_mode", "auto"))
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.inference_mode_var, 
                                 values=["auto", "cpu", "gpu", "best_performance"], state="readonly")
        mode_combo.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        mode_combo.bind('<<ComboboxSelected>>', self.on_inference_mode_changed)
        
        # 执行提供者
        provider_frame = ttk.Frame(inference_frame)
        provider_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(provider_frame, text="执行提供者:").pack(side=tk.LEFT)
        self.provider_var = tk.StringVar()
        self.provider_combo = ttk.Combobox(provider_frame, textvariable=self.provider_var, 
                                          values=self.device_manager.available_providers, state="readonly")
        self.provider_combo.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        self.provider_combo.bind('<<ComboboxSelected>>', self.on_provider_changed)
        
        # 更新当前提供者
        current_providers = self.config.get("providers", [])
        if current_providers:
            self.provider_var.set(current_providers[0])
    
    def create_detection_config(self, parent):
        """创建检测配置"""
        detection_frame = ttk.Frame(parent)
        detection_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(detection_frame, text="检测参数", style='Subtitle.TLabel').pack(anchor=tk.W)
        
        # 置信度阈值
        conf_frame = ttk.Frame(detection_frame)
        conf_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(conf_frame, text="置信度阈值:").pack(side=tk.LEFT)
        self.conf_threshold_var = tk.DoubleVar(value=self.config.get("conf_threshold", 0.4))
        conf_scale = ttk.Scale(conf_frame, from_=0.1, to=0.9, variable=self.conf_threshold_var, 
                              orient=tk.HORIZONTAL, length=200)
        conf_scale.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        self.conf_label = ttk.Label(conf_frame, text=f"{self.conf_threshold_var.get():.2f}")
        self.conf_label.pack(side=tk.RIGHT)
        conf_scale.configure(command=self.on_conf_changed)
        
        # NMS阈值
        nms_frame = ttk.Frame(detection_frame)
        nms_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(nms_frame, text="NMS阈值:").pack(side=tk.LEFT)
        self.nms_threshold_var = tk.DoubleVar(value=self.config.get("nms_threshold", 0.4))
        nms_scale = ttk.Scale(nms_frame, from_=0.1, to=0.9, variable=self.nms_threshold_var, 
                             orient=tk.HORIZONTAL, length=200)
        nms_scale.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        self.nms_label = ttk.Label(nms_frame, text=f"{self.nms_threshold_var.get():.2f}")
        self.nms_label.pack(side=tk.RIGHT)
        nms_scale.configure(command=self.on_nms_changed)
        
        # 性能监控开关
        perf_frame = ttk.Frame(detection_frame)
        perf_frame.pack(fill=tk.X, pady=5)
        
        self.performance_monitoring_var = tk.BooleanVar(value=self.config.get("performance_monitoring", True))
        ttk.Checkbutton(perf_frame, text="启用性能监控", 
                       variable=self.performance_monitoring_var,
                       command=self.on_performance_monitoring_changed).pack(side=tk.LEFT)
    
    def create_device_info(self, parent):
        """创建设备信息显示"""
        # 设备摘要
        summary_frame = ttk.Frame(parent)
        summary_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(summary_frame, text="检测到的设备", style='Subtitle.TLabel').pack(anchor=tk.W)
        
        # 创建设备信息文本框
        self.device_info_text = scrolledtext.ScrolledText(summary_frame, height=8, width=50)
        self.device_info_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 刷新按钮
        ttk.Button(summary_frame, text="刷新设备信息", command=self.refresh_device_info).pack(pady=5)
    
    def create_control_buttons(self, parent):
        """创建控制按钮"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X)
        
        # 启动/停止按钮
        self.start_button = ttk.Button(button_frame, text="启动程序", 
                                      command=self.start_assistant, style='Primary.TButton')
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="停止程序", 
                                     command=self.stop_assistant, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # 保存配置按钮
        ttk.Button(button_frame, text="保存配置", command=self.save_config).pack(side=tk.LEFT, padx=(0, 5))
        
        # 重置配置按钮
        ttk.Button(button_frame, text="重置配置", command=self.reset_config).pack(side=tk.LEFT)
    
    def create_right_panel(self, parent):
        """创建右侧面板"""
        # 性能监控部分
        perf_frame = ttk.LabelFrame(parent, text="性能监控", padding="10")
        perf_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.create_performance_monitor(perf_frame)
        
        # 日志显示部分
        log_frame = ttk.LabelFrame(parent, text="运行日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.create_log_display(log_frame)
    
    def create_performance_monitor(self, parent):
        """创建性能监控界面"""
        # 性能数据显示
        stats_frame = ttk.Frame(parent)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 当前性能
        current_frame = ttk.LabelFrame(stats_frame, text="当前性能")
        current_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.fps_label = ttk.Label(current_frame, text="FPS: --", style='Info.TLabel')
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        self.latency_label = ttk.Label(current_frame, text="延迟: --ms", style='Info.TLabel')
        self.latency_label.pack(side=tk.LEFT, padx=5)
        
        self.memory_label = ttk.Label(current_frame, text="内存: --MB", style='Info.TLabel')
        self.memory_label.pack(side=tk.LEFT, padx=5)
        
        # 性能图表
        chart_frame = ttk.Frame(parent)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图表
        self.fig = Figure(figsize=(8, 6), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 控制按钮
        chart_control_frame = ttk.Frame(parent)
        chart_control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(chart_control_frame, text="刷新图表", command=self.update_performance_chart).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(chart_control_frame, text="重置统计", command=self.reset_performance_stats).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(chart_control_frame, text="导出数据", command=self.export_performance_data).pack(side=tk.LEFT)
        
        # 启动性能更新定时器
        self.update_performance_display()
    
    def create_log_display(self, parent):
        """创建日志显示"""
        self.log_text = scrolledtext.ScrolledText(parent, height=15, width=70)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # 日志控制
        log_control_frame = ttk.Frame(parent)
        log_control_frame.pack(fill=tk.X)
        
        ttk.Button(log_control_frame, text="清空日志", command=self.clear_log).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(log_control_frame, text="保存日志", command=self.save_log).pack(side=tk.LEFT)
        
        # 设置日志处理器
        self.setup_log_handler()
    
    def setup_log_handler(self):
        """设置日志处理器"""
        class GUILogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
                
            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.text_widget.insert(tk.END, msg + '\n')
                    self.text_widget.see(tk.END)
                    
                    # 限制日志行数
                    lines = int(self.text_widget.index('end-1c').split('.')[0])
                    if lines > 1000:
                        self.text_widget.delete('1.0', '500.0')
                except:
                    pass
        
        handler = GUILogHandler(self.log_text)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(handler)
    
    def refresh_model_list(self, combo):
        """刷新模型列表"""
        try:
            model_dir = "onnxmd"
            if os.path.exists(model_dir):
                models = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
                combo['values'] = models
                if models and not self.model_list_var.get():
                    self.model_list_var.set(models[0])
        except Exception as e:
            logger.error(f"刷新模型列表失败: {e}")
    
    def refresh_device_info(self):
        """刷新设备信息"""
        try:
            summary = self.device_manager.get_device_summary()
            
            info_text = "=== 设备检测摘要 ===\n\n"
            info_text += f"推荐提供者: {summary['recommended_provider']}\n"
            info_text += f"可用提供者: {', '.join(summary['available_providers'])}\n\n"
            
            info_text += "=== 检测到的设备 ===\n"
            for device_type, device_info in summary['devices'].items():
                info_text += f"\n{device_type}:\n"
                if device_type == "CPU":
                    info_text += f"  核心数: {device_info['cores']}\n"
                    info_text += f"  架构: {device_info['architecture']}\n"
                else:
                    info_text += f"  设备数量: {device_info['count']}\n"
                    for i, device in enumerate(device_info['devices']):
                        info_text += f"  设备{i+1}: {device}\n"
                    
                    if device_type == "AMD" and 'rocm_support' in device_info:
                        info_text += f"  ROCm支持: {'是' if device_info['rocm_support'] else '否'}\n"
                    elif device_type == "NVIDIA":
                        if 'cuda_version' in device_info and device_info['cuda_version']:
                            info_text += f"  CUDA版本: {device_info['cuda_version']}\n"
                        if 'driver_version' in device_info and device_info['driver_version']:
                            info_text += f"  驱动版本: {device_info['driver_version']}\n"
                    elif device_type == "Intel" and 'openvino_version' in device_info:
                        if device_info['openvino_version']:
                            info_text += f"  OpenVINO版本: {device_info['openvino_version']}\n"
            
            self.device_info_text.delete('1.0', tk.END)
            self.device_info_text.insert('1.0', info_text)
            
        except Exception as e:
            logger.error(f"刷新设备信息失败: {e}")
            self.device_info_text.delete('1.0', tk.END)
            self.device_info_text.insert('1.0', f"刷新设备信息失败: {e}")
    
    def update_performance_display(self):
        """更新性能显示"""
        try:
            if self.is_running and self.config.get("performance_monitoring", False):
                # 获取当前提供者
                current_provider = self.provider_var.get()
                if current_provider:
                    recent_perf = self.performance_monitor.get_recent_performance(current_provider)
                    
                    if 'recent_fps' in recent_perf:
                        self.fps_label.config(text=f"FPS: {recent_perf['recent_fps']:.1f}")
                    
                    if 'avg_inference_time' in recent_perf:
                        latency_ms = recent_perf['avg_inference_time'] * 1000
                        self.latency_label.config(text=f"延迟: {latency_ms:.1f}ms")
                    
                    if 'avg_memory_usage' in recent_perf:
                        self.memory_label.config(text=f"内存: {recent_perf['avg_memory_usage']:.1f}MB")
            
        except Exception as e:
            logger.debug(f"更新性能显示失败: {e}")
        
        # 每秒更新一次
        self.root.after(1000, self.update_performance_display)
    
    def update_performance_chart(self):
        """更新性能图表"""
        try:
            self.fig.clear()
            
            all_stats = self.performance_monitor.get_all_stats()
            if not all_stats:
                return
            
            # 创建子图
            ax1 = self.fig.add_subplot(221)  # FPS
            ax2 = self.fig.add_subplot(222)  # 延迟
            ax3 = self.fig.add_subplot(223)  # 内存使用
            ax4 = self.fig.add_subplot(224)  # 对比
            
            providers = list(all_stats.keys())
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
            
            # FPS图表
            fps_values = [stats.get('fps', {}).get('avg', 0) for stats in all_stats.values()]
            if any(fps_values):
                bars1 = ax1.bar(providers, fps_values, color=colors[:len(providers)])
                ax1.set_title('平均FPS')
                ax1.set_ylabel('FPS')
                ax1.tick_params(axis='x', rotation=45)
            
            # 延迟图表
            latency_values = [stats.get('inference_time', {}).get('avg', 0) * 1000 for stats in all_stats.values()]
            if any(latency_values):
                bars2 = ax2.bar(providers, latency_values, color=colors[:len(providers)])
                ax2.set_title('平均延迟')
                ax2.set_ylabel('毫秒')
                ax2.tick_params(axis='x', rotation=45)
            
            # 内存使用图表
            memory_values = [stats.get('memory_usage', {}).get('avg', 0) for stats in all_stats.values()]
            if any(memory_values):
                bars3 = ax3.bar(providers, memory_values, color=colors[:len(providers)])
                ax3.set_title('平均内存使用')
                ax3.set_ylabel('MB')
                ax3.tick_params(axis='x', rotation=45)
            
            # 综合对比（归一化）
            if fps_values and latency_values:
                # 归一化分数（FPS越高越好，延迟越低越好）
                max_fps = max(fps_values) if max(fps_values) > 0 else 1
                min_latency = min([l for l in latency_values if l > 0]) if any(l > 0 for l in latency_values) else 1
                
                scores = []
                for fps, latency in zip(fps_values, latency_values):
                    fps_score = fps / max_fps
                    latency_score = min_latency / latency if latency > 0 else 0
                    combined_score = (fps_score + latency_score) / 2
                    scores.append(combined_score)
                
                bars4 = ax4.bar(providers, scores, color=colors[:len(providers)])
                ax4.set_title('综合性能评分')
                ax4.set_ylabel('评分')
                ax4.set_ylim(0, 1)
                ax4.tick_params(axis='x', rotation=45)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            logger.error(f"更新性能图表失败: {e}")
    
    # 事件处理方法
    def browse_model_file(self):
        """浏览模型文件"""
        filename = filedialog.askopenfilename(
            title="选择ONNX模型文件",
            filetypes=[("ONNX files", "*.onnx"), ("All files", "*.*")],
            initialdir="onnxmd"
        )
        if filename:
            self.model_path_var.set(filename)
            self.config["model_path"] = filename
    
    def on_model_selected(self, event):
        """模型选择事件"""
        selected_model = self.model_list_var.get()
        if selected_model:
            model_path = os.path.join("onnxmd", selected_model)
            self.model_path_var.set(model_path)
            self.config["model_path"] = model_path
    
    def on_inference_mode_changed(self, event):
        """推理模式改变事件"""
        mode = self.inference_mode_var.get()
        self.config["inference_mode"] = mode
        
        # 根据模式更新提供者
        if mode == "auto":
            provider = self.device_manager.recommended_provider
        elif mode == "cpu":
            provider = "CPUExecutionProvider"
        elif mode == "gpu":
            gpu_providers = [p for p in self.device_manager.available_providers 
                           if p != "CPUExecutionProvider"]
            provider = gpu_providers[0] if gpu_providers else "CPUExecutionProvider"
        else:  # best_performance
            provider = self.device_manager.recommended_provider
        
        self.provider_var.set(provider)
        self.config["providers"] = [provider]
        self.config["provider_options"] = [self.device_manager.get_provider_config(provider)]
    
    def on_provider_changed(self, event):
        """提供者改变事件"""
        provider = self.provider_var.get()
        self.config["providers"] = [provider]
        self.config["provider_options"] = [self.device_manager.get_provider_config(provider)]
        self.config["inference_mode"] = "custom"
        self.inference_mode_var.set("custom")
    
    def on_conf_changed(self, value):
        """置信度阈值改变事件"""
        val = float(value)
        self.conf_label.config(text=f"{val:.2f}")
        self.config["conf_threshold"] = val
    
    def on_nms_changed(self, value):
        """NMS阈值改变事件"""
        val = float(value)
        self.nms_label.config(text=f"{val:.2f}")
        self.config["nms_threshold"] = val
    
    def on_performance_monitoring_changed(self):
        """性能监控开关改变事件"""
        self.config["performance_monitoring"] = self.performance_monitoring_var.get()
    
    def start_assistant(self):
        """启动游戏辅助程序"""
        try:
            if self.is_running:
                return
            
            # 保存当前配置
            self.save_config()
            
            # 启动助手线程
            def run_assistant():
                try:
                    # 导入并运行主程序
                    import main
                    assistant = main.GameAssistant()
                    assistant.run()
                except Exception as e:
                    logger.error(f"运行游戏辅助程序失败: {e}")
                finally:
                    self.is_running = False
                    self.root.after(0, self.update_control_buttons)
            
            self.assistant_thread = threading.Thread(target=run_assistant, daemon=True)
            self.assistant_thread.start()
            
            self.is_running = True
            self.update_control_buttons()
            
            logger.info("游戏辅助程序已启动")
            
        except Exception as e:
            logger.error(f"启动游戏辅助程序失败: {e}")
            messagebox.showerror("错误", f"启动失败: {e}")
    
    def stop_assistant(self):
        """停止游戏辅助程序"""
        try:
            self.is_running = False
            self.update_control_buttons()
            logger.info("游戏辅助程序已停止")
        except Exception as e:
            logger.error(f"停止游戏辅助程序失败: {e}")
    
    def update_control_buttons(self):
        """更新控制按钮状态"""
        if self.is_running:
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        else:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def reset_config(self):
        """重置配置"""
        if messagebox.askyesno("确认", "确定要重置所有配置吗？"):
            self.config = self.get_default_config()
            self.save_config()
            
            # 更新界面
            self.model_path_var.set(self.config["model_path"])
            self.inference_mode_var.set(self.config["inference_mode"])
            self.provider_var.set(self.config["providers"][0])
            self.conf_threshold_var.set(self.config["conf_threshold"])
            self.nms_threshold_var.set(self.config["nms_threshold"])
            self.performance_monitoring_var.set(self.config["performance_monitoring"])
            
            logger.info("配置已重置")
    
    def reset_performance_stats(self):
        """重置性能统计"""
        self.performance_monitor.reset_stats()
        self.update_performance_chart()
        logger.info("性能统计已重置")
    
    def export_performance_data(self):
        """导出性能数据"""
        try:
            filename = filedialog.asksaveasfilename(
                title="保存性能数据",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                data = self.performance_monitor.export_stats()
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False, default=str)
                logger.info(f"性能数据已导出到: {filename}")
        except Exception as e:
            logger.error(f"导出性能数据失败: {e}")
            messagebox.showerror("错误", f"导出失败: {e}")
    
    def clear_log(self):
        """清空日志"""
        self.log_text.delete('1.0', tk.END)
    
    def save_log(self):
        """保存日志"""
        try:
            filename = filedialog.asksaveasfilename(
                title="保存日志文件",
                defaultextension=".log",
                filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get('1.0', tk.END))
                logger.info(f"日志已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存日志失败: {e}")
            messagebox.showerror("错误", f"保存失败: {e}")
    
    def on_closing(self):
        """关闭窗口事件"""
        if self.is_running:
            if messagebox.askyesno("确认", "程序正在运行，确定要退出吗？"):
                self.stop_assistant()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """主函数"""
    root = tk.Tk()
    app = EnhancedGameAssistantGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()