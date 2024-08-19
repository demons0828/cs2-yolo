import time
import numpy as np
import onnxruntime as ort
from PIL import ImageGrab, Image
import cv2
import pyautogui
import pynput.keyboard as keyboard
from pynput.mouse import Controller
from move import simulate_mouse_movement

# 类别名称和置信度阈值
names = {0: 't_body', 1: 't_head', 2: 'ct_body', 3: 'ct_head'}
conf_threshold = 0.4
nms_threshold = 0.4
mouse = Controller()

# 图像预处理和推理相关参数
inputtag = 'input'
input_size = (640, 640)  # 输入图像大小
screen_crop_size = (1920, 1080)  # 截取屏幕区域大小
# 获取屏幕中心点
screen_width, screen_height = pyautogui.size()
# 定义 screen_center 作为常量
SCREEN_CENTER = (screen_width // 2, screen_height // 2)
# 加载ONNX模型
onnx_model_path = r'yolo\yoloaimonnx\onnxmd\10w320v5.onnx'
providers = ['ROCMExecutionProvider']
ort_session = ort.InferenceSession(onnx_model_path)

# 定义一个标志来控制循环
running = True
t_or_ct = True
show_image = True  # 新增变量，控制是否显示图像和绘制框

def preprocess(image):
    image = image.resize(input_size)
    image = np.array(image).astype(np.float32)
    image = image / 255.0  # 归一化
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # 增加batch维度
    return image

def postprocess(outputs):
    boxes, scores, class_ids = [], [], []
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
    
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    result_boxes, result_scores, result_class_ids = [], [], []
    for i in indices:
        result_boxes.append(boxes[i])
        result_scores.append(scores[i])
        result_class_ids.append(class_ids[i])
    
    return result_boxes, result_scores, result_class_ids

def draw_boxes(image, boxes, scores, class_ids):
    """
    image 是 PIL.Image 类型
    boxes 是检测框的坐标列表
    """
    h, w, _ = image.shape
    scale_x = w / input_size[0]
    scale_y = h / input_size[1]
    for box, score, class_id in zip(boxes, scores, class_ids):
        x, y, box_w, box_h = box
        x1, y1 = int((x - box_w / 2) * scale_x), int((y - box_h / 2) * scale_y)
        x2, y2 = int((x + box_w / 2) * scale_x), int((y + box_h / 2) * scale_y)
        label = f"{names[class_id]}: {score:.2f}"
        if class_id == 0 or class_id == 1:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def aim_head(boxes, scores, class_ids, screen_center=SCREEN_CENTER, scale_x=screen_crop_size[0]/input_size[0], scale_y=screen_crop_size[1]/input_size[1]):
    head_boxes = []
    head_scores = []
    head_class_ids = []

    for box, score, class_id in zip(boxes, scores, class_ids):
        if (class_id == 1 and t_or_ct) or (class_id == 3 and not t_or_ct):  # 只选择头部目标
            head_boxes.append(box)
            head_scores.append(score)
            head_class_ids.append(class_id)
            if not head_boxes:
                return  # 没有检测到头部目标
    if not head_boxes:
        return

    # 计算头部目标与屏幕中心的距离
    distances = []
    for box in head_boxes:
        x, y, box_w, box_h = box
        head_center_x = x
        head_center_y = y
        distance = np.sqrt((head_center_x - screen_center[0]) ** 2 + (head_center_y - screen_center[1]) ** 2)
        distances.append(distance)

    # 找到最近的头部目标
    min_distance_index = np.argmin(distances)
    closest_head_box = head_boxes[min_distance_index]

    # 计算最近头部目标的中心点
    x, y, box_w, box_h = closest_head_box
    head_center_x = int(x * scale_x + screen_center[0] - screen_crop_size[0] // 2)
    head_center_y = int(y * scale_y + screen_center[1] - screen_crop_size[1] // 2)
    # 获取鼠标当前位置
    mouse_x, mouse_y = mouse.position

    # 定义一个范围（例如，头部目标的中心点的一个小范围）
    range_x1 = head_center_x - box_w * scale_x // 2
    range_x2 = head_center_x + box_w * scale_x // 2
    range_y1 = head_center_y - box_h * scale_y // 2
    range_y2 = head_center_y + box_h * scale_y // 2

    # 检查鼠标是否在范围内
    if range_x1 <= mouse_x <= range_x2 and range_y1 <= mouse_y <= range_y2:
        simulate_mouse_movement((head_center_x, head_center_y), mouse)
    else:
        mouse.position = (head_center_x, head_center_y)

def on_press(key):
    global running, t_or_ct, show_image
    try:
        if key.char == 'e': # 移动鼠标到最近的头部目标
            aim_head(boxes, scores, class_ids)
        elif key.char == 'q': # 退出程序
            running = False
            print('退出程序')
        elif key.char == 'o': # 切换 t 和 ct 的状态
            t_or_ct = not t_or_ct
            print('当前标签为:', 't' if t_or_ct else 'ct')
        elif key.char == 'i':
            show_image = not show_image
            print('显示图像:', show_image)
            # 切换显示图像和绘制框的状态
    except AttributeError:
        pass

# 启动键盘监听器
listener = keyboard.Listener(on_press=on_press)
listener.start()

while running:
    start_time = time.time()
    screen = ImageGrab.grab()
    screen_width, screen_height = screen.size
    left = (screen_width - screen_crop_size[0]) // 2
    top = (screen_height - screen_crop_size[1]) // 2
    right = left + screen_crop_size[0]
    bottom = top + screen_crop_size[1]
    pm_center = screen.crop((left, top, right, bottom))

    input_image = preprocess(pm_center)
    outputs = ort_session.run(None, {inputtag: input_image})
    boxes, scores, class_ids = postprocess(outputs)

    if show_image:
        pm_center = np.array(pm_center)
        draw_boxes(pm_center, boxes, scores, class_ids)
        # 将图像从RGB转换为BGR
        pm_center = cv2.cvtColor(pm_center, cv2.COLOR_RGB2BGR)
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(pm_center, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 创建一个可调整大小的窗口
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        # 设置窗口大小，例如宽度为800，高度为600
        cv2.resizeWindow('Detection', screen_crop_size[0]//3, screen_crop_size[1]//3)
        
        cv2.imshow('Detection', pm_center)
    cv2.waitKey(1)

cv2.destroyAllWindows()
listener.stop()