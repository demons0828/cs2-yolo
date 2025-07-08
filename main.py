import time
import numpy as np
import onnxruntime as ort
from PIL import ImageGrab, Image
import cv2
import pyautogui
import pynput.keyboard as keyboard
from pynput.mouse import Controller
from move import simulate_mouse_movement
import os
import sys
import logging
from typing import List, Tuple, Optional, Dict, Any
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """配置管理类"""
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.default_config = {
            "model_path": "yolo/yoloaimonnx/onnxmd/10w320v5.onnx",
            "conf_threshold": 0.4,
            "nms_threshold": 0.4,
            "input_size": [640, 640],
            "screen_crop_size": [1920, 1080],
            "providers": ["ROCMExecutionProvider"],
            "names": {0: 't_body', 1: 't_head', 2: 'ct_body', 3: 'ct_head'}
        }
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    logger.info(f"配置文件加载成功: {self.config_file}")
                    return config
            else:
                self.save_config(self.default_config)
                logger.info(f"创建默认配置文件: {self.config_file}")
                return self.default_config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self.default_config
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")

class YOLODetector:
    """YOLO检测器类"""
    def __init__(self, config: Config):
        self.config = config
        self.ort_session = None
        self.input_tag = 'input'
        self.initialize_model()
    
    def initialize_model(self) -> None:
        """初始化ONNX模型"""
        try:
            model_path = self.config.config["model_path"]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            providers = self.config.config["providers"]
            self.ort_session = ort.InferenceSession(model_path, providers=providers)
            logger.info(f"模型加载成功: {model_path}")
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """图像预处理"""
        try:
            input_size = tuple(self.config.config["input_size"])
            image = image.resize(input_size)
            image = np.array(image).astype(np.float32)
            image = image / 255.0  # 归一化
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            image = np.expand_dims(image, axis=0)  # 增加batch维度
            return image
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            raise
    
    def postprocess(self, outputs: List[np.ndarray]) -> Tuple[List, List, List]:
        """后处理检测结果"""
        try:
            boxes, scores, class_ids = [], [], []
            conf_threshold = self.config.config["conf_threshold"]
            nms_threshold = self.config.config["nms_threshold"]
            
            for output in outputs[0][0]:
                if len(output) < 6:
                    continue
                box = output[:4]
                score = output[4]
                class_id = np.argmax(output[5:])
                if score > conf_threshold:
                    boxes.append(box)
                    scores.append(score)
                    class_ids.append(class_id)
            
            if not boxes:
                return [], [], []
            
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
            if len(indices) == 0:
                return [], [], []
            
            result_boxes, result_scores, result_class_ids = [], [], []
            for i in indices.flatten():
                result_boxes.append(boxes[i])
                result_scores.append(scores[i])
                result_class_ids.append(class_ids[i])
            
            return result_boxes, result_scores, result_class_ids
        except Exception as e:
            logger.error(f"后处理失败: {e}")
            return [], [], []
    
    def detect(self, image: Image.Image) -> Tuple[List, List, List]:
        """执行检测"""
        try:
            input_image = self.preprocess(image)
            outputs = self.ort_session.run(None, {self.input_tag: input_image})
            return self.postprocess(outputs)
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return [], [], []

class ScreenCapture:
    """屏幕捕获类"""
    def __init__(self, config: Config):
        self.config = config
        self.screen_width, self.screen_height = pyautogui.size()
        self.screen_center = (self.screen_width // 2, self.screen_height // 2)
        self.screen_crop_size = tuple(self.config.config["screen_crop_size"])
    
    def capture_screen_center(self) -> Optional[Image.Image]:
        """捕获屏幕中心区域"""
        try:
            screen = ImageGrab.grab()
            screen_width, screen_height = screen.size
            
            left = (screen_width - self.screen_crop_size[0]) // 2
            top = (screen_height - self.screen_crop_size[1]) // 2
            right = left + self.screen_crop_size[0]
            bottom = top + self.screen_crop_size[1]
            
            return screen.crop((left, top, right, bottom))
        except Exception as e:
            logger.error(f"屏幕捕获失败: {e}")
            return None

class AimController:
    """瞄准控制器类"""
    def __init__(self, config: Config, mouse: Controller):
        self.config = config
        self.mouse = mouse
        self.screen_center = (pyautogui.size()[0] // 2, pyautogui.size()[1] // 2)
        self.screen_crop_size = tuple(self.config.config["screen_crop_size"])
        self.input_size = tuple(self.config.config["input_size"])
        self.names = self.config.config["names"]
    
    def aim_head(self, boxes: List, scores: List, class_ids: List, t_or_ct: bool) -> None:
        """瞄准头部目标"""
        try:
            head_boxes = []
            head_scores = []
            head_class_ids = []

            for box, score, class_id in zip(boxes, scores, class_ids):
                if (class_id == 1 and t_or_ct) or (class_id == 3 and not t_or_ct):
                    head_boxes.append(box)
                    head_scores.append(score)
                    head_class_ids.append(class_id)
            
            if not head_boxes:
                return

            # 计算头部目标与屏幕中心的距离
            distances = []
            for box in head_boxes:
                x, y, box_w, box_h = box
                head_center_x = x
                head_center_y = y
                distance = np.sqrt((head_center_x - self.screen_center[0]) ** 2 + 
                                 (head_center_y - self.screen_center[1]) ** 2)
                distances.append(distance)

            # 找到最近的头部目标
            min_distance_index = np.argmin(distances)
            closest_head_box = head_boxes[min_distance_index]

            # 计算最近头部目标的中心点
            x, y, box_w, box_h = closest_head_box
            scale_x = self.screen_crop_size[0] / self.input_size[0]
            scale_y = self.screen_crop_size[1] / self.input_size[1]
            
            head_center_x = int(x * scale_x + self.screen_center[0] - self.screen_crop_size[0] // 2)
            head_center_y = int(y * scale_y + self.screen_center[1] - self.screen_crop_size[1] // 2)
            
            # 获取鼠标当前位置
            mouse_x, mouse_y = self.mouse.position

            # 定义瞄准范围
            range_x1 = head_center_x - box_w * scale_x // 2
            range_x2 = head_center_x + box_w * scale_x // 2
            range_y1 = head_center_y - box_h * scale_y // 2
            range_y2 = head_center_y + box_h * scale_y // 2

            # 检查鼠标是否在范围内
            if range_x1 <= mouse_x <= range_x2 and range_y1 <= mouse_y <= range_y2:
                simulate_mouse_movement((head_center_x, head_center_y), self.mouse)
            else:
                self.mouse.position = (head_center_x, head_center_y)
                
        except Exception as e:
            logger.error(f"瞄准失败: {e}")

class Visualizer:
    """可视化类"""
    def __init__(self, config: Config):
        self.config = config
        self.input_size = tuple(self.config.config["input_size"])
        self.names = self.config.config["names"]
    
    def draw_boxes(self, image: np.ndarray, boxes: List, scores: List, class_ids: List) -> np.ndarray:
        """绘制检测框"""
        try:
            h, w, _ = image.shape
            scale_x = w / self.input_size[0]
            scale_y = h / self.input_size[1]
            
            for box, score, class_id in zip(boxes, scores, class_ids):
                x, y, box_w, box_h = box
                x1, y1 = int((x - box_w / 2) * scale_x), int((y - box_h / 2) * scale_y)
                x2, y2 = int((x + box_w / 2) * scale_x), int((y + box_h / 2) * scale_y)
                label = f"{self.names[class_id]}: {score:.2f}"
                
                if class_id == 0 or class_id == 1:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return image
        except Exception as e:
            logger.error(f"绘制检测框失败: {e}")
            return image

class GameAssistant:
    """游戏辅助主类"""
    def __init__(self):
        self.config = Config()
        self.detector = YOLODetector(self.config)
        self.screen_capture = ScreenCapture(self.config)
        self.aim_controller = AimController(self.config, Controller())
        self.visualizer = Visualizer(self.config)
        
        self.running = True
        self.t_or_ct = True
        self.show_image = True
        
        # 键盘监听器
        self.listener = None
        self.setup_keyboard_listener()
    
    def setup_keyboard_listener(self) -> None:
        """设置键盘监听器"""
        try:
            self.listener = keyboard.Listener(on_press=self.on_press)
            self.listener.start()
            logger.info("键盘监听器启动成功")
        except Exception as e:
            logger.error(f"键盘监听器启动失败: {e}")
    
    def on_press(self, key) -> None:
        """键盘按键处理"""
        try:
            if hasattr(key, 'char'):
                if key.char == 'e':
                    # 移动鼠标到最近的头部目标
                    if hasattr(self, 'last_boxes'):
                        self.aim_controller.aim_head(self.last_boxes, self.last_scores, self.last_class_ids, self.t_or_ct)
                elif key.char == 'q':
                    # 退出程序
                    self.running = False
                    logger.info('程序退出')
                elif key.char == 'o':
                    # 切换 t 和 ct 的状态
                    self.t_or_ct = not self.t_or_ct
                    logger.info(f'当前标签为: {"t" if self.t_or_ct else "ct"}')
                elif key.char == 'i':
                    # 切换显示图像状态
                    self.show_image = not self.show_image
                    logger.info(f'显示图像: {self.show_image}')
        except Exception as e:
            logger.error(f"键盘事件处理失败: {e}")
    
    def run(self) -> None:
        """主运行循环"""
        try:
            logger.info("游戏辅助程序启动")
            
            while self.running:
                start_time = time.time()
                
                # 捕获屏幕
                screen_center = self.screen_capture.capture_screen_center()
                if screen_center is None:
                    logger.warning("屏幕捕获失败，跳过本次循环")
                    continue
                
                # 执行检测
                boxes, scores, class_ids = self.detector.detect(screen_center)
                
                # 保存检测结果供键盘事件使用
                self.last_boxes = boxes
                self.last_scores = scores
                self.last_class_ids = class_ids
                
                # 可视化
                if self.show_image:
                    self.display_detection(screen_center, boxes, scores, class_ids, start_time)
                
                # 控制显示频率
                cv2.waitKey(1)
                
        except KeyboardInterrupt:
            logger.info("程序被用户中断")
        except Exception as e:
            logger.error(f"主循环运行失败: {e}")
        finally:
            self.cleanup()
    
    def display_detection(self, screen_center: Image.Image, boxes: List, scores: List, class_ids: List, start_time: float) -> None:
        """显示检测结果"""
        try:
            # 转换为numpy数组
            image = np.array(screen_center)
            
            # 绘制检测框
            image = self.visualizer.draw_boxes(image, boxes, scores, class_ids)
            
            # 转换为BGR格式
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 计算并显示FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 创建可调整大小的窗口
            cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Detection', self.config.config["screen_crop_size"][0]//3, 
                           self.config.config["screen_crop_size"][1]//3)
            
            cv2.imshow('Detection', image)
            
        except Exception as e:
            logger.error(f"显示检测结果失败: {e}")
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.listener:
                self.listener.stop()
            cv2.destroyAllWindows()
            logger.info("资源清理完成")
        except Exception as e:
            logger.error(f"资源清理失败: {e}")

def main():
    """主函数"""
    try:
        assistant = GameAssistant()
        assistant.run()
    except Exception as e:
        logger.error(f"程序启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()