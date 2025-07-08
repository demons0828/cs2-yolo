import numpy as np
from pynput.mouse import Controller
import time
import matplotlib.pyplot as plt
import logging
from typing import Tuple, List, Optional
import sys

# 配置日志
logger = logging.getLogger(__name__)

class MouseMovementSimulator:
    """鼠标移动模拟器类"""
    
    def __init__(self, mouse: Controller):
        """
        初始化鼠标移动模拟器
        
        Args:
            mouse: 鼠标控制器实例
        """
        self.mouse = mouse
        self.default_params = {
            'jindu': 1000,      # 轨迹点数量
            'noise_level': 0.58, # 噪声水平
            'speed': 0.00001,    # 移动速度
            'show_image': False  # 是否显示轨迹图
        }
    
    def _validate_point(self, point: Tuple[int, int]) -> bool:
        """
        验证坐标点是否有效
        
        Args:
            point: 坐标点 (x, y)
            
        Returns:
            bool: 坐标点是否有效
        """
        try:
            x, y = point
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return False
            if x < 0 or y < 0:
                return False
            return True
        except (TypeError, ValueError):
            return False
    
    def _generate_random_control_points(self, P0: np.ndarray, P3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成贝塞尔曲线的随机控制点
        
        Args:
            P0: 起点坐标
            P3: 终点坐标
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 控制点P1和P2
        """
        try:
            x_min, x_max = min(P0[0], P3[0]), max(P0[0], P3[0])
            y_min, y_max = min(P0[1], P3[1]), max(P0[1], P3[1])
            
            # 添加一些随机偏移以避免直线
            offset_x = (x_max - x_min) * 0.1
            offset_y = (y_max - y_min) * 0.1
            
            P1 = np.array([
                np.random.uniform(x_min - offset_x, x_max + offset_x),
                np.random.uniform(y_min - offset_y, y_max + offset_y)
            ])
            P2 = np.array([
                np.random.uniform(x_min - offset_x, x_max + offset_x),
                np.random.uniform(y_min - offset_y, y_max + offset_y)
            ])
            
            return P1, P2
        except Exception as e:
            logger.error(f"生成控制点失败: {e}")
            # 返回默认控制点
            return P0 + (P3 - P0) * 0.25, P0 + (P3 - P0) * 0.75
    
    def _bezier_curve(self, t: float, P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, P3: np.ndarray) -> np.ndarray:
        """
        计算贝塞尔曲线上的点
        
        Args:
            t: 参数值 (0-1)
            P0, P1, P2, P3: 贝塞尔曲线的控制点
            
        Returns:
            np.ndarray: 曲线上的点坐标
        """
        try:
            return ((1 - t)**3 * P0 + 
                    3 * (1 - t)**2 * t * P1 + 
                    3 * (1 - t) * t**2 * P2 + 
                    t**3 * P3)
        except Exception as e:
            logger.error(f"贝塞尔曲线计算失败: {e}")
            return P0 + t * (P3 - P0)  # 回退到线性插值
    
    def _add_noise(self, points: np.ndarray, noise_level: float) -> np.ndarray:
        """
        为轨迹点添加噪声
        
        Args:
            points: 原始轨迹点
            noise_level: 噪声水平
            
        Returns:
            np.ndarray: 添加噪声后的轨迹点
        """
        try:
            if noise_level <= 0:
                return points
            
            noise = np.random.normal(0, noise_level, points.shape)
            noisy_points = points + noise
            
            # 确保噪声不会使坐标变为负数
            noisy_points = np.maximum(noisy_points, 0)
            
            return noisy_points
        except Exception as e:
            logger.error(f"添加噪声失败: {e}")
            return points
    
    def _plot_trajectory(self, trajectory: List[Tuple[float, float]]) -> None:
        """
        绘制鼠标移动轨迹
        
        Args:
            trajectory: 轨迹点列表
        """
        try:
            if not trajectory:
                logger.warning("轨迹为空，无法绘制")
                return
            
            x_coords = [point[0] for point in trajectory]
            y_coords = [point[1] for point in trajectory]
            
            plt.ion()  # 打开交互模式
            plt.figure(figsize=(10, 8))
            plt.plot(x_coords, y_coords, marker='o', markersize=2, alpha=0.7)
            plt.title('Mouse Movement Trajectory with Noise')
            plt.xlabel('X Coordinates')
            plt.ylabel('Y Coordinates')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.draw()
            plt.pause(0.001)
            
        except Exception as e:
            logger.error(f"绘制轨迹失败: {e}")
    
    def simulate_movement(self, 
                         end_point: Tuple[int, int], 
                         jindu: Optional[int] = None,
                         noise_level: Optional[float] = None, 
                         speed: Optional[float] = None, 
                         show_image: Optional[bool] = None) -> bool:
        """
        模拟鼠标平滑移动
        
        Args:
            end_point: 目标终点坐标
            jindu: 轨迹点数量
            noise_level: 噪声水平
            speed: 移动速度
            show_image: 是否显示轨迹图
            
        Returns:
            bool: 移动是否成功
        """
        try:
            # 使用默认参数或传入的参数
            jindu = jindu if jindu is not None else self.default_params['jindu']
            noise_level = noise_level if noise_level is not None else self.default_params['noise_level']
            speed = speed if speed is not None else self.default_params['speed']
            show_image = show_image if show_image is not None else self.default_params['show_image']
            
            # 参数验证
            if not self._validate_point(end_point):
                logger.error(f"无效的终点坐标: {end_point}")
                return False
            
            if jindu <= 0:
                logger.error(f"无效的轨迹点数量: {jindu}")
                return False
            
            if noise_level < 0:
                logger.error(f"无效的噪声水平: {noise_level}")
                return False
            
            if speed <= 0:
                logger.error(f"无效的移动速度: {speed}")
                return False
            
            # 获取起点和终点
            P0 = np.array(self.mouse.position)
            P3 = np.array(end_point)
            
            # 检查起点和终点是否相同
            if np.allclose(P0, P3, atol=1):
                logger.info("起点和终点相同，无需移动")
                return True
            
            # 生成控制点
            P1, P2 = self._generate_random_control_points(P0, P3)
            
            # 生成贝塞尔曲线点
            t_values = np.linspace(0, 1, jindu)
            curve_points = np.array([self._bezier_curve(t, P0, P1, P2, P3) for t in t_values])
            
            # 添加噪声
            noisy_curve_points = self._add_noise(curve_points, noise_level)
            
            # 保存移动轨迹
            trajectory = []
            
            # 执行鼠标移动
            start_time = time.time()
            try:
                for i, point in enumerate(noisy_curve_points):
                    # 确保坐标是整数
                    x, y = int(round(point[0])), int(round(point[1]))
                    
                    # 移动鼠标
                    self.mouse.position = (x, y)
                    trajectory.append((x, y))
                    
                    # 控制移动速度
                    if i < len(noisy_curve_points) - 1:  # 不是最后一个点
                        end_time = time.perf_counter() + speed
                        while time.perf_counter() < end_time:
                            pass
                
                end_time = time.time()
                logger.info(f"鼠标移动完成，耗时: {end_time - start_time:.2f}s")
                
                # 显示轨迹图
                if show_image:
                    self._plot_trajectory(trajectory)
                
                return True
                
            except Exception as e:
                logger.error(f"鼠标移动过程中出错: {e}")
                return False
                
        except Exception as e:
            logger.error(f"模拟鼠标移动失败: {e}")
            return False
    
    def move_to_point(self, point: Tuple[int, int]) -> bool:
        """
        直接移动到指定点（无动画）
        
        Args:
            point: 目标坐标
            
        Returns:
            bool: 移动是否成功
        """
        try:
            if not self._validate_point(point):
                logger.error(f"无效的坐标点: {point}")
                return False
            
            self.mouse.position = point
            logger.info(f"鼠标已移动到: {point}")
            return True
            
        except Exception as e:
            logger.error(f"直接移动失败: {e}")
            return False

# 为了保持向后兼容性，保留原来的函数接口
def simulate_mouse_movement(end_point: Tuple[int, int], 
                          mouse: Controller, 
                          jindu: int = 1000, 
                          noise_level: float = 0.58, 
                          speed: float = 0.00001, 
                          show_image: bool = False) -> bool:
    """
    模拟鼠标移动的兼容性函数
    
    Args:
        end_point: 目标终点坐标
        mouse: 鼠标控制器
        jindu: 轨迹点数量
        noise_level: 噪声水平
        speed: 移动速度
        show_image: 是否显示轨迹图
        
    Returns:
        bool: 移动是否成功
    """
    try:
        simulator = MouseMovementSimulator(mouse)
        return simulator.simulate_movement(end_point, jindu, noise_level, speed, show_image)
    except Exception as e:
        logger.error(f"鼠标移动失败: {e}")
        return False

if __name__ == '__main__':
    # 测试代码
    try:
        mouse = Controller()
        simulator = MouseMovementSimulator(mouse)
        
        # 测试直接移动
        print("测试直接移动...")
        success = simulator.move_to_point((500, 300))
        print(f"直接移动结果: {success}")
        
        # 测试平滑移动
        print("测试平滑移动...")
        success = simulator.simulate_movement((800, 600), show_image=True)
        print(f"平滑移动结果: {success}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        sys.exit(1)