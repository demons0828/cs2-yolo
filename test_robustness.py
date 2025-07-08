#!/usr/bin/env python3
"""
健壮性测试脚本
用于验证改进后代码的稳定性和错误处理能力
"""

import sys
import os
import time
import threading
import logging
from typing import List, Tuple
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from error_handler import ErrorHandler, ResourceManager, ThreadSafeCounter
from move import MouseMovementSimulator
from main import Config, YOLODetector, ScreenCapture, AimController, Visualizer

# 配置测试日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestErrorHandler(unittest.TestCase):
    """错误处理器测试类"""
    
    def setUp(self):
        self.error_handler = ErrorHandler()
    
    def test_exception_handling(self):
        """测试异常处理"""
        try:
            raise ValueError("测试异常")
        except Exception as e:
            self.error_handler.handle_exception(e, "测试上下文")
        
        self.assertEqual(self.error_handler.get_error_count(), 1)
    
    def test_retry_mechanism(self):
        """测试重试机制"""
        call_count = 0
        
        @self.error_handler.retry_on_failure(max_retries=2, delay=0.1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("模拟失败")
            return "成功"
        
        result = failing_function()
        self.assertEqual(result, "成功")
        self.assertEqual(call_count, 3)
    
    def test_safe_execute(self):
        """测试安全执行"""
        def test_func(x, y):
            return x + y
        
        result = self.error_handler.safe_execute(test_func, 2, 3)
        self.assertEqual(result, 5)
        
        # 测试异常情况
        def failing_func():
            raise ValueError("测试错误")
        
        result = self.error_handler.safe_execute(failing_func)
        self.assertIsNone(result)

class TestResourceManager(unittest.TestCase):
    """资源管理器测试类"""
    
    def setUp(self):
        self.resource_manager = ResourceManager()
        self.cleanup_called = False
    
    def test_resource_registration(self):
        """测试资源注册"""
        def cleanup_func(resource):
            self.cleanup_called = True
        
        mock_resource = Mock()
        self.resource_manager.register_resource(mock_resource, cleanup_func)
        
        self.resource_manager.cleanup()
        self.assertTrue(self.cleanup_called)
    
    def test_cleanup_callback(self):
        """测试清理回调"""
        callback_called = False
        
        def cleanup_callback():
            nonlocal callback_called
            callback_called = True
        
        self.resource_manager.register_cleanup_callback(cleanup_callback)
        self.resource_manager.cleanup()
        
        self.assertTrue(callback_called)

class TestThreadSafeCounter(unittest.TestCase):
    """线程安全计数器测试类"""
    
    def setUp(self):
        self.counter = ThreadSafeCounter()
    
    def test_basic_operations(self):
        """测试基本操作"""
        self.assertEqual(self.counter.get_value(), 0)
        
        self.counter.increment(5)
        self.assertEqual(self.counter.get_value(), 5)
        
        self.counter.decrement(2)
        self.assertEqual(self.counter.get_value(), 3)
        
        self.counter.reset()
        self.assertEqual(self.counter.get_value(), 0)
    
    def test_thread_safety(self):
        """测试线程安全性"""
        def increment_worker():
            for _ in range(100):
                self.counter.increment(1)
        
        def decrement_worker():
            for _ in range(100):
                self.counter.decrement(1)
        
        # 创建多个线程
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=increment_worker))
            threads.append(threading.Thread(target=decrement_worker))
        
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证最终结果
        self.assertEqual(self.counter.get_value(), 0)

class TestMouseMovementSimulator(unittest.TestCase):
    """鼠标移动模拟器测试类"""
    
    def setUp(self):
        self.mock_mouse = Mock()
        self.mock_mouse.position = (100, 100)
        self.simulator = MouseMovementSimulator(self.mock_mouse)
    
    def test_point_validation(self):
        """测试坐标点验证"""
        # 有效坐标
        self.assertTrue(self.simulator._validate_point((100, 200)))
        self.assertTrue(self.simulator._validate_point((0, 0)))
        
        # 无效坐标
        self.assertFalse(self.simulator._validate_point((-1, 100)))
        self.assertFalse(self.simulator._validate_point((100, -1)))
    
    def test_bezier_curve(self):
        """测试贝塞尔曲线计算"""
        P0 = np.array([0, 0])
        P1 = np.array([1, 1])
        P2 = np.array([2, 1])
        P3 = np.array([3, 0])
        
        # 测试起点
        result = self.simulator._bezier_curve(0, P0, P1, P2, P3)
        np.testing.assert_array_almost_equal(result, P0)
        
        # 测试终点
        result = self.simulator._bezier_curve(1, P0, P1, P2, P3)
        np.testing.assert_array_almost_equal(result, P3)
    
    def test_noise_addition(self):
        """测试噪声添加"""
        points = np.array([[1, 2], [3, 4], [5, 6]])
        
        # 无噪声
        result = self.simulator._add_noise(points, 0)
        np.testing.assert_array_equal(result, points)
        
        # 有噪声
        result = self.simulator._add_noise(points, 0.1)
        self.assertEqual(result.shape, points.shape)
        # 确保噪声不会使坐标变为负数
        self.assertTrue(np.all(result >= 0))

class TestConfig(unittest.TestCase):
    """配置管理测试类"""
    
    def setUp(self):
        self.test_config_file = "test_config.json"
        self.config = Config(self.test_config_file)
    
    def tearDown(self):
        # 清理测试文件
        if os.path.exists(self.test_config_file):
            os.remove(self.test_config_file)
    
    def test_config_loading(self):
        """测试配置加载"""
        # 测试默认配置
        self.assertIn("model_path", self.config.config)
        self.assertIn("conf_threshold", self.config.config)
        self.assertIn("names", self.config.config)
    
    def test_config_saving(self):
        """测试配置保存"""
        test_config = {"test_key": "test_value"}
        self.config.save_config(test_config)
        
        # 验证文件是否创建
        self.assertTrue(os.path.exists(self.test_config_file))

def run_stress_test():
    """运行压力测试"""
    logger.info("开始压力测试...")
    
    # 创建错误处理器
    error_handler = ErrorHandler()
    
    # 模拟大量异常
    for i in range(100):
        try:
            if i % 10 == 0:
                raise ValueError(f"测试异常 {i}")
        except Exception as e:
            error_handler.handle_exception(e, f"压力测试 {i}")
    
    logger.info(f"压力测试完成，总错误数: {error_handler.get_error_count()}")
    
    # 测试线程安全计数器
    counter = ThreadSafeCounter()
    
    def worker():
        for _ in range(1000):
            counter.increment(1)
            time.sleep(0.001)
    
    threads = [threading.Thread(target=worker) for _ in range(10)]
    
    start_time = time.time()
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    logger.info(f"线程安全测试完成，最终值: {counter.get_value()}, 耗时: {end_time - start_time:.2f}s")

def run_integration_test():
    """运行集成测试"""
    logger.info("开始集成测试...")
    
    try:
        # 测试配置加载
        config = Config()
        logger.info("配置加载测试通过")
        
        # 测试资源管理器
        resource_manager = ResourceManager()
        
        def cleanup_callback():
            logger.info("清理回调执行")
        
        resource_manager.register_cleanup_callback(cleanup_callback)
        resource_manager.cleanup()
        logger.info("资源管理器测试通过")
        
        # 测试错误处理器
        error_handler = ErrorHandler()
        
        @error_handler.retry_on_failure(max_retries=1, delay=0.1)
        def test_function():
            return "集成测试成功"
        
        result = test_function()
        logger.info(f"错误处理器测试通过: {result}")
        
        logger.info("集成测试完成")
        
    except Exception as e:
        logger.error(f"集成测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # 运行单元测试
    logger.info("运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行压力测试
    logger.info("\n" + "="*50)
    run_stress_test()
    
    # 运行集成测试
    logger.info("\n" + "="*50)
    success = run_integration_test()
    
    if success:
        logger.info("所有测试通过！代码健壮性良好。")
    else:
        logger.error("部分测试失败，需要进一步检查。")
        sys.exit(1)