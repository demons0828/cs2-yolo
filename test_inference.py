#!/usr/bin/env python3
"""
CS2 YOLO模型推理测试脚本
使用CS2游戏截图测试ONNX模型的目标检测能力
"""

import os
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import logging
import glob
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NAMES = {0: 't_body', 1: 't_head', 2: 'ct_body', 3: 'ct_head'}
COLORS = {
    0: (0, 0, 255),    # t_body - 红色
    1: (0, 165, 255),  # t_head - 橙色
    2: (0, 255, 0),    # ct_body - 绿色
    3: (255, 255, 0),  # ct_head - 青色
}


def load_model(model_path: str) -> ort.InferenceSession:
    """加载ONNX模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    logger.info(f"模型加载成功: {model_path}")
    input_info = session.get_inputs()[0]
    logger.info(f"  输入: {input_info.name}, shape={input_info.shape}")
    for out in session.get_outputs():
        logger.info(f"  输出: {out.name}, shape={out.shape}")
    return session


def preprocess(image: Image.Image, input_size=(640, 640)) -> np.ndarray:
    """图像预处理"""
    image = image.convert('RGB')
    image = image.resize(input_size)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


def postprocess_v5(outputs, conf_threshold=0.25, nms_threshold=0.4):
    """YOLOv5后处理 (10w320v5.onnx: output shape [1, 25200, 9])"""
    boxes, scores, class_ids = [], [], []

    for det in outputs[0][0]:
        if len(det) < 6:
            continue
        box = det[:4].tolist()
        obj_conf = float(det[4])
        cls_scores = det[5:]
        class_id = int(np.argmax(cls_scores))
        cls_conf = float(cls_scores[class_id])
        score = obj_conf * cls_conf

        if score > conf_threshold:
            boxes.append(box)
            scores.append(score)
            class_ids.append(class_id)

    if not boxes:
        return [], [], []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    if len(indices) == 0:
        return [], [], []

    result_boxes = [boxes[i] for i in indices.flatten()]
    result_scores = [scores[i] for i in indices.flatten()]
    result_class_ids = [class_ids[i] for i in indices.flatten()]

    return result_boxes, result_scores, result_class_ids


def postprocess_v8(outputs, conf_threshold=0.25, nms_threshold=0.4):
    """YOLOv8后处理 (best1.onnx: output shape [1, 8, 8400])"""
    output = outputs[0]  # [1, 8, 8400]
    output = output[0].T  # [8400, 8]

    boxes, scores, class_ids = [], [], []

    for det in output:
        box = det[:4].tolist()
        cls_scores = det[4:]
        class_id = int(np.argmax(cls_scores))
        score = float(cls_scores[class_id])

        if score > conf_threshold:
            boxes.append(box)
            scores.append(score)
            class_ids.append(class_id)

    if not boxes:
        return [], [], []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    if len(indices) == 0:
        return [], [], []

    result_boxes = [boxes[i] for i in indices.flatten()]
    result_scores = [scores[i] for i in indices.flatten()]
    result_class_ids = [class_ids[i] for i in indices.flatten()]

    return result_boxes, result_scores, result_class_ids


def draw_detections(image: np.ndarray, boxes, scores, class_ids, input_size=(640, 640)):
    """在图像上绘制检测结果"""
    h, w, _ = image.shape
    scale_x = w / input_size[0]
    scale_y = h / input_size[1]

    for box, score, class_id in zip(boxes, scores, class_ids):
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * scale_x)
        y1 = int((cy - bh / 2) * scale_y)
        x2 = int((cx + bw / 2) * scale_x)
        y2 = int((cy + bh / 2) * scale_y)

        color = COLORS.get(class_id, (255, 255, 255))
        label = f"{NAMES.get(class_id, f'cls{class_id}')}: {score:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def run_inference(session, image_path, postprocess_fn, conf_threshold=0.25):
    """对单张图片运行推理"""
    logger.info(f"处理图片: {image_path}")

    img = Image.open(image_path).convert('RGB')
    orig_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    input_data = preprocess(img)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})

    boxes, scores, class_ids = postprocess_fn(outputs, conf_threshold=conf_threshold)

    logger.info(f"  检测到 {len(boxes)} 个目标:")
    for box, score, cid in zip(boxes, scores, class_ids):
        logger.info(f"    {NAMES.get(cid, f'cls{cid}')}: {score:.3f} @ [{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

    result_image = draw_detections(orig_image.copy(), boxes, scores, class_ids)
    return result_image, len(boxes)


def main():
    output_dir = "/workspace/test_results"
    os.makedirs(output_dir, exist_ok=True)

    image_dir = "/workspace/test_images"
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    if not image_files:
        logger.error(f"未找到测试图片: {image_dir}")
        sys.exit(1)

    logger.info(f"找到 {len(image_files)} 张测试图片")

    models = [
        ("onnxmd/10w320v5.onnx", "v5", postprocess_v5),
        ("onnxmd/best1.onnx", "v8", postprocess_v8),
    ]

    total_detections = 0

    for model_path, model_tag, postprocess_fn in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"模型: {model_path} ({model_tag})")
        logger.info(f"{'='*60}")

        session = load_model(model_path)

        for img_path in image_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            result_image, num_det = run_inference(session, img_path, postprocess_fn, conf_threshold=0.25)
            total_detections += num_det

            out_path = os.path.join(output_dir, f"{basename}_{model_tag}_result.jpg")
            cv2.imwrite(out_path, result_image)
            logger.info(f"  结果保存: {out_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"推理完成！总共检测到 {total_detections} 个目标")
    logger.info(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
