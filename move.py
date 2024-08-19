import numpy as np
from pynput.mouse import Controller
import time
import matplotlib.pyplot as plt

def simulate_mouse_movement(end_point,mouse,jindu=1000,noise_level=0.58, speed=0.00001, show_image=False):
    # # 创建一个 Controller 实例
    # mouse = Controller()
    
    # 获取鼠标当前位置作为起点
    P0 = np.array(mouse.position)  # 起点
    P3 = np.array(end_point)  # 终点

    # 随机生成控制点P1和P2
    def generate_random_control_points(P0, P3):
        x_min, x_max = min(P0[0], P3[0]), max(P0[0], P3[0])
        y_min, y_max = min(P0[1], P3[1]), max(P0[1], P3[1])
        P1 = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])
        P2 = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])
        return P1, P2

    P1, P2 = generate_random_control_points(P0, P3)

    # 贝塞尔曲线函数
    def bezier_curve(t, P0, P1, P2, P3):
        return (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3

    # 生成贝塞尔曲线点
    t_values = np.linspace(0, 1, jindu)
    curve_points = np.array([bezier_curve(t, P0, P1, P2, P3) for t in t_values])

    # 添加随机噪声
    noisy_curve_points = curve_points + np.random.normal(0, noise_level, curve_points.shape)

    # 保存移动轨迹
    trajectory = []

    # 模拟鼠标平滑移动
    t0 = time.time()
    for point in noisy_curve_points:
        mouse.position = (point[0], point[1])
        trajectory.append((point[0], point[1]))
        end_time = time.perf_counter() + speed  # 控制移动速度
        while time.perf_counter() < end_time:
            pass
    t1 = time.time()
    print(f"Total time: {t1 - t0:.2f}s")
    if show_image:
        
        # 生成图像
        x_coords = [point[0] for point in trajectory]
        y_coords = [point[1] for point in trajectory]
        plt.ion()  # 打开交互模式
        plt.plot(x_coords, y_coords, marker='o')
        plt.title('Mouse Movement Trajectory with Noise')
        plt.xlabel('X Coordinates')
        plt.ylabel('Y Coordinates')
        plt.grid(True)
        plt.draw()  # 更新图形
        plt.pause(0.001)  # 短暂暂停以更新图形
if __name__ == '__main__':
    # 调用函数
    # simulate_mouse_movement([500, 100])
    pass