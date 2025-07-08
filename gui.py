import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import logging
from typing import Dict, Any, Optional, Tuple
from pynput.mouse import Controller
from move import MouseMovementSimulator
import os
import sys

class BezierCurveGUI:
    """贝塞尔曲线鼠标移动模拟GUI界面"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("贝塞尔曲线鼠标移动模拟器")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # 初始化组件
        self.mouse = Controller()
        self.simulator = MouseMovementSimulator(self.mouse)
        self.config = self.load_config()
        self.is_simulating = False
        self.trajectory_data = []
        
        # 设置样式
        self.setup_styles()
        
        # 创建界面
        self.create_widgets()
        
        # 绑定事件
        self.bind_events()
        
        # 更新显示
        self.update_mouse_position()
        
    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Info.TLabel', font=('Arial', 10))
        
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 确保mouse_movement配置存在
                    if 'mouse_movement' not in config:
                        config['mouse_movement'] = {
                            'jindu': 1000,
                            'noise_level': 0.58,
                            'speed': 0.00001,
                            'show_trajectory': False
                        }
                    return config
            else:
                return {
                    'mouse_movement': {
                        'jindu': 1000,
                        'noise_level': 0.58,
                        'speed': 0.00001,
                        'show_trajectory': False
                    }
                }
        except Exception as e:
            messagebox.showerror("错误", f"加载配置文件失败: {e}")
            return {
                'mouse_movement': {
                    'jindu': 1000,
                    'noise_level': 0.58,
                    'speed': 0.00001,
                    'show_trajectory': False
                }
            }
    
    def save_config(self):
        """保存配置文件"""
        try:
            # 更新配置
            self.config['mouse_movement'] = {
                'jindu': self.jindu_var.get(),
                'noise_level': self.noise_level_var.get(),
                'speed': self.speed_var.get(),
                'show_trajectory': self.show_trajectory_var.get()
            }
            
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            
            messagebox.showinfo("成功", "配置已保存")
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {e}")
    
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
        title_label = ttk.Label(main_frame, text="贝塞尔曲线鼠标移动模拟器", style='Title.TLabel')
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
        
        # 参数设置
        params_frame = ttk.LabelFrame(control_frame, text="参数设置", padding="5")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 轨迹点数量
        ttk.Label(params_frame, text="轨迹点数量:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.jindu_var = tk.IntVar(value=self.config['mouse_movement']['jindu'])
        jindu_scale = ttk.Scale(params_frame, from_=100, to=2000, variable=self.jindu_var, orient=tk.HORIZONTAL)
        jindu_scale.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=2)
        self.jindu_label = ttk.Label(params_frame, text=str(self.jindu_var.get()))
        self.jindu_label.grid(row=0, column=2, padx=(5, 0), pady=2)
        
        # 噪声水平
        ttk.Label(params_frame, text="噪声水平:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.noise_level_var = tk.DoubleVar(value=self.config['mouse_movement']['noise_level'])
        noise_scale = ttk.Scale(params_frame, from_=0.0, to=2.0, variable=self.noise_level_var, orient=tk.HORIZONTAL)
        noise_scale.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=2)
        self.noise_label = ttk.Label(params_frame, text=f"{self.noise_level_var.get():.2f}")
        self.noise_label.grid(row=1, column=2, padx=(5, 0), pady=2)
        
        # 移动速度
        ttk.Label(params_frame, text="移动速度:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.speed_var = tk.DoubleVar(value=self.config['mouse_movement']['speed'])
        speed_scale = ttk.Scale(params_frame, from_=0.000001, to=0.0001, variable=self.speed_var, orient=tk.HORIZONTAL)
        speed_scale.grid(row=2, column=1, sticky="ew", padx=(5, 0), pady=2)
        self.speed_label = ttk.Label(params_frame, text=f"{self.speed_var.get():.6f}")
        self.speed_label.grid(row=2, column=2, padx=(5, 0), pady=2)
        
        # 显示轨迹
        self.show_trajectory_var = tk.BooleanVar(value=self.config['mouse_movement']['show_trajectory'])
        ttk.Checkbutton(params_frame, text="显示轨迹图", variable=self.show_trajectory_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # 配置网格权重
        params_frame.columnconfigure(1, weight=1)
        
        # 目标坐标设置
        target_frame = ttk.LabelFrame(control_frame, text="目标坐标", padding="5")
        target_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(target_frame, text="X坐标:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.target_x_var = tk.IntVar(value=500)
        ttk.Entry(target_frame, textvariable=self.target_x_var, width=10).grid(row=0, column=1, padx=(5, 0), pady=2)
        
        ttk.Label(target_frame, text="Y坐标:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.target_y_var = tk.IntVar(value=300)
        ttk.Entry(target_frame, textvariable=self.target_y_var, width=10).grid(row=1, column=1, padx=(5, 0), pady=2)
        
        # 快速坐标按钮
        quick_frame = ttk.Frame(target_frame)
        quick_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Button(quick_frame, text="屏幕中心", command=self.set_screen_center).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_frame, text="随机位置", command=self.set_random_position).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_frame, text="获取当前", command=self.get_current_position).pack(side=tk.LEFT, padx=2)
        
        # 操作按钮
        action_frame = ttk.LabelFrame(control_frame, text="操作", padding="5")
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.simulate_btn = ttk.Button(action_frame, text="开始模拟", command=self.start_simulation)
        self.simulate_btn.pack(fill=tk.X, pady=2)
        
        self.stop_btn = ttk.Button(action_frame, text="停止模拟", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="直接移动", command=self.direct_move).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="保存配置", command=self.save_config).pack(fill=tk.X, pady=2)
        
        # 日志显示
        log_frame = ttk.LabelFrame(control_frame, text="日志", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=40)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 绑定参数变化事件
        jindu_scale.configure(command=self.on_jindu_change)
        noise_scale.configure(command=self.on_noise_change)
        speed_scale.configure(command=self.on_speed_change)
    
    def create_display_panel(self, parent):
        """创建右侧显示面板"""
        display_frame = ttk.LabelFrame(parent, text="轨迹显示", padding="10")
        display_frame.grid(row=1, column=1, sticky="nsew")
        
        # 创建matplotlib图形
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化图形
        self.update_plot()
    
    def create_status_bar(self, parent):
        """创建底部状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        # 鼠标位置显示
        self.mouse_pos_label = ttk.Label(status_frame, text="鼠标位置: (0, 0)", style='Info.TLabel')
        self.mouse_pos_label.pack(side=tk.LEFT)
        
        # 状态显示
        self.status_label = ttk.Label(status_frame, text="就绪", style='Info.TLabel')
        self.status_label.pack(side=tk.RIGHT)
    
    def bind_events(self):
        """绑定事件"""
        # 窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 鼠标移动事件
        self.root.bind('<Motion>', self.on_mouse_move)
    
    def on_jindu_change(self, value):
        """轨迹点数量变化事件"""
        self.jindu_label.config(text=str(int(float(value))))
    
    def on_noise_change(self, value):
        """噪声水平变化事件"""
        self.noise_label.config(text=f"{float(value):.2f}")
    
    def on_speed_change(self, value):
        """移动速度变化事件"""
        self.speed_label.config(text=f"{float(value):.6f}")
    
    def on_mouse_move(self, event):
        """鼠标移动事件"""
        self.update_mouse_position()
    
    def update_mouse_position(self):
        """更新鼠标位置显示"""
        try:
            pos = self.mouse.position
            self.mouse_pos_label.config(text=f"鼠标位置: ({pos[0]}, {pos[1]})")
        except Exception as e:
            self.log_message(f"获取鼠标位置失败: {e}")
    
    def set_screen_center(self):
        """设置屏幕中心坐标"""
        try:
            import pyautogui
            width, height = pyautogui.size()
            self.target_x_var.set(width // 2)
            self.target_y_var.set(height // 2)
            self.log_message(f"设置目标坐标: 屏幕中心 ({width // 2}, {height // 2})")
        except Exception as e:
            self.log_message(f"设置屏幕中心失败: {e}")
    
    def set_random_position(self):
        """设置随机坐标"""
        try:
            import pyautogui
            width, height = pyautogui.size()
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            self.target_x_var.set(x)
            self.target_y_var.set(y)
            self.log_message(f"设置目标坐标: 随机位置 ({x}, {y})")
        except Exception as e:
            self.log_message(f"设置随机位置失败: {e}")
    
    def get_current_position(self):
        """获取当前鼠标位置"""
        try:
            pos = self.mouse.position
            self.target_x_var.set(pos[0])
            self.target_y_var.set(pos[1])
            self.log_message(f"获取当前鼠标位置: ({pos[0]}, {pos[1]})")
        except Exception as e:
            self.log_message(f"获取当前位置失败: {e}")
    
    def start_simulation(self):
        """开始模拟"""
        if self.is_simulating:
            return
        
        target_x = self.target_x_var.get()
        target_y = self.target_y_var.get()
        
        if target_x < 0 or target_y < 0:
            messagebox.showerror("错误", "目标坐标不能为负数")
            return
        
        self.is_simulating = True
        self.simulate_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="模拟中...")
        
        # 在新线程中运行模拟
        thread = threading.Thread(target=self.run_simulation, args=(target_x, target_y))
        thread.daemon = True
        thread.start()
    
    def run_simulation(self, target_x: int, target_y: int):
        """运行模拟"""
        try:
            self.log_message(f"开始模拟移动到 ({target_x}, {target_y})")
            
            # 获取参数
            jindu = self.jindu_var.get()
            noise_level = self.noise_level_var.get()
            speed = self.speed_var.get()
            show_trajectory = self.show_trajectory_var.get()
            
            # 记录开始位置
            start_pos = self.mouse.position
            self.trajectory_data = [(start_pos[0], start_pos[1])]
            
            # 执行模拟
            success, trajectory = self.simulator.simulate_movement(
                (target_x, target_y),
                jindu=jindu,
                noise_level=noise_level,
                speed=speed,
                show_image=show_trajectory
            )
            
            if success:
                self.log_message("模拟完成")
                # 更新轨迹数据
                self.trajectory_data = trajectory
                self.update_trajectory_plot()
            else:
                self.log_message("模拟失败")
            
        except Exception as e:
            self.log_message(f"模拟出错: {e}")
        finally:
            # 在主线程中更新UI
            self.root.after(0, self.simulation_finished)
    
    def simulation_finished(self):
        """模拟完成后的UI更新"""
        self.is_simulating = False
        self.simulate_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="就绪")
    
    def stop_simulation(self):
        """停止模拟"""
        self.is_simulating = False
        self.simulate_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="已停止")
        self.log_message("模拟已停止")
    
    def direct_move(self):
        """直接移动"""
        try:
            target_x = self.target_x_var.get()
            target_y = self.target_y_var.get()
            
            if target_x < 0 or target_y < 0:
                messagebox.showerror("错误", "目标坐标不能为负数")
                return
            
            success = self.simulator.move_to_point((target_x, target_y))
            
            if success:
                self.log_message(f"直接移动到 ({target_x}, {target_y})")
                self.update_mouse_position()
            else:
                self.log_message("直接移动失败")
                
        except Exception as e:
            self.log_message(f"直接移动出错: {e}")
    
    def update_trajectory_plot(self):
        """更新轨迹图"""
        try:
            if not self.trajectory_data:
                return
            
            # 清除当前图形
            self.ax.clear()
            
            # 提取坐标
            x_coords = [point[0] for point in self.trajectory_data]
            y_coords = [point[1] for point in self.trajectory_data]
            
            # 绘制轨迹
            self.ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='移动轨迹')
            
            # 标记起点和终点
            if len(self.trajectory_data) > 0:
                start_point = self.trajectory_data[0]
                end_point = self.trajectory_data[-1]
                
                self.ax.plot(start_point[0], start_point[1], 'go', markersize=8, label='起点')
                self.ax.plot(end_point[0], end_point[1], 'ro', markersize=8, label='终点')
            
            # 设置图形属性
            self.ax.set_xlabel('X坐标')
            self.ax.set_ylabel('Y坐标')
            self.ax.set_title('鼠标移动轨迹')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            
            # 显示屏幕边界
            try:
                import pyautogui
                width, height = pyautogui.size()
                self.ax.set_xlim(0, width)
                self.ax.set_ylim(height, 0)  # 反转Y轴以匹配屏幕坐标
            except:
                self.ax.set_xlim(0, 1920)
                self.ax.set_ylim(1080, 0)
            
            # 更新画布
            self.canvas.draw()
            
        except Exception as e:
            self.log_message(f"更新轨迹图失败: {e}")
    
    def update_plot(self):
        """更新图形显示"""
        try:
            self.ax.clear()
            self.ax.set_xlabel('X坐标')
            self.ax.set_ylabel('Y坐标')
            self.ax.set_title('鼠标移动轨迹')
            self.ax.grid(True, alpha=0.3)
            
            # 显示屏幕边界
            try:
                import pyautogui
                width, height = pyautogui.size()
                self.ax.set_xlim(0, width)
                self.ax.set_ylim(height, 0)  # 反转Y轴以匹配屏幕坐标
            except:
                self.ax.set_xlim(0, 1920)
                self.ax.set_ylim(1080, 0)
            
            self.canvas.draw()
        except Exception as e:
            self.log_message(f"更新图形失败: {e}")
    
    def log_message(self, message: str):
        """添加日志消息"""
        try:
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            # 在主线程中更新UI
            self.root.after(0, lambda: self.log_text.insert(tk.END, log_entry))
            self.root.after(0, lambda: self.log_text.see(tk.END))
        except Exception as e:
            print(f"日志记录失败: {e}")
    
    def on_closing(self):
        """窗口关闭事件"""
        if self.is_simulating:
            if messagebox.askokcancel("退出", "模拟正在进行中，确定要退出吗？"):
                self.stop_simulation()
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = BezierCurveGUI(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("错误", f"启动GUI失败: {e}")

if __name__ == "__main__":
    main()