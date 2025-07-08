#!/usr/bin/env python3
"""
简化健壮性测试脚本
只测试基本的错误处理功能，不依赖外部库
"""

import sys
import os
import time
import threading
import logging
from typing import List, Tuple, Optional, Dict, Any
from functools import wraps
import json

# 配置测试日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorHandler:
    """错误处理工具类"""
    
    def __init__(self):
        self.error_count = 0
        self.max_retries = 3
        self.retry_delay = 1.0
        self.error_callbacks: Dict[str, Any] = {}
    
    def handle_exception(self, exception: Exception, context: str = "") -> None:
        """处理异常并记录日志"""
        self.error_count += 1
        error_msg = f"错误 #{self.error_count} - {context}: {str(exception)}"
        logger.info(error_msg)
        
        # 调用错误回调函数
        if context in self.error_callbacks:
            try:
                self.error_callbacks[context](exception)
            except Exception as e:
                logger.error(f"错误回调函数执行失败: {e}")
    
    def retry_on_failure(self, max_retries: Optional[int] = None, delay: Optional[float] = None):
        """装饰器：在失败时重试"""
        max_retries = max_retries if max_retries is not None else self.max_retries
        delay = delay if delay is not None else self.retry_delay
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries:
                            logger.info(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                            logger.info(f"等待 {delay} 秒后重试...")
                            time.sleep(delay)
                        else:
                            logger.error(f"函数 {func.__name__} 在 {max_retries + 1} 次尝试后仍然失败")
                
                # 所有重试都失败了
                if last_exception is not None:
                    raise last_exception
                else:
                    raise RuntimeError("未知错误")
            
            return wrapper
        return decorator
    
    def safe_execute(self, func, *args, **kwargs):
        """安全执行函数，捕获所有异常"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_exception(e, f"执行函数 {func.__name__}")
            return None
    
    def register_error_callback(self, context: str, callback) -> None:
        """注册错误回调函数"""
        self.error_callbacks[context] = callback
    
    def reset_error_count(self) -> None:
        """重置错误计数"""
        self.error_count = 0
    
    def get_error_count(self) -> int:
        """获取错误计数"""
        return self.error_count

class ResourceManager:
    """资源管理器，确保资源正确释放"""
    
    def __init__(self):
        self.resources = []
        self.cleanup_callbacks = []
    
    def register_resource(self, resource: Any, cleanup_func) -> None:
        """注册需要清理的资源"""
        self.resources.append((resource, cleanup_func))
    
    def register_cleanup_callback(self, callback) -> None:
        """注册清理回调函数"""
        self.cleanup_callbacks.append(callback)
    
    def cleanup(self) -> None:
        """清理所有资源"""
        logger.info("开始清理资源...")
        
        # 执行清理回调
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"清理回调执行失败: {e}")
        
        # 清理注册的资源
        for resource, cleanup_func in self.resources:
            try:
                cleanup_func(resource)
            except Exception as e:
                logger.error(f"资源清理失败: {e}")
        
        self.resources.clear()
        self.cleanup_callbacks.clear()
        logger.info("资源清理完成")

class ThreadSafeCounter:
    """线程安全计数器"""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        """增加计数"""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """减少计数"""
        with self._lock:
            self._value -= amount
            return self._value
    
    def get_value(self) -> int:
        """获取当前值"""
        with self._lock:
            return self._value
    
    def reset(self) -> None:
        """重置计数器"""
        with self._lock:
            self._value = 0

class Config:
    """配置管理类"""
    def __init__(self, config_file: str = "test_config.json"):
        self.config_file = config_file
        self.default_config = {
            "model_path": "test_model.onnx",
            "conf_threshold": 0.4,
            "nms_threshold": 0.4,
            "input_size": [640, 640],
            "screen_crop_size": [1920, 1080],
            "providers": ["CPUExecutionProvider"],
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

def test_error_handler():
    """测试错误处理器"""
    logger.info("测试错误处理器...")
    
    error_handler = ErrorHandler()
    
    # 测试异常处理
    try:
        raise ValueError("测试异常")
    except Exception as e:
        error_handler.handle_exception(e, "测试上下文")
    
    assert error_handler.get_error_count() == 1
    logger.info("✓ 异常处理测试通过")
    
    # 测试重试机制
    call_count = 0
    
    @error_handler.retry_on_failure(max_retries=2, delay=0.1)
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("模拟失败")
        return "成功"
    
    result = failing_function()
    assert result == "成功"
    assert call_count == 3
    logger.info("✓ 重试机制测试通过")
    
    # 测试安全执行
    def test_func(x, y):
        return x + y
    
    result = error_handler.safe_execute(test_func, 2, 3)
    assert result == 5
    logger.info("✓ 安全执行测试通过")
    
    # 测试异常情况
    def failing_func():
        raise ValueError("测试错误")
    
    result = error_handler.safe_execute(failing_func)
    assert result is None
    logger.info("✓ 异常安全执行测试通过")

def test_resource_manager():
    """测试资源管理器"""
    logger.info("测试资源管理器...")
    
    resource_manager = ResourceManager()
    cleanup_called = False
    
    def cleanup_func(resource):
        nonlocal cleanup_called
        cleanup_called = True
    
    mock_resource = "test_resource"
    resource_manager.register_resource(mock_resource, cleanup_func)
    
    resource_manager.cleanup()
    assert cleanup_called
    logger.info("✓ 资源注册测试通过")
    
    # 测试清理回调
    callback_called = False
    
    def cleanup_callback():
        nonlocal callback_called
        callback_called = True
    
    resource_manager.register_cleanup_callback(cleanup_callback)
    resource_manager.cleanup()
    
    assert callback_called
    logger.info("✓ 清理回调测试通过")

def test_thread_safe_counter():
    """测试线程安全计数器"""
    logger.info("测试线程安全计数器...")
    
    counter = ThreadSafeCounter()
    
    # 测试基本操作
    assert counter.get_value() == 0
    
    counter.increment(5)
    assert counter.get_value() == 5
    
    counter.decrement(2)
    assert counter.get_value() == 3
    
    counter.reset()
    assert counter.get_value() == 0
    logger.info("✓ 基本操作测试通过")
    
    # 测试线程安全性
    def increment_worker():
        for _ in range(100):
            counter.increment(1)
    
    def decrement_worker():
        for _ in range(100):
            counter.decrement(1)
    
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
    assert counter.get_value() == 0
    logger.info("✓ 线程安全测试通过")

def test_config():
    """测试配置管理"""
    logger.info("测试配置管理...")
    
    test_config_file = "test_config.json"
    config = Config(test_config_file)
    
    # 测试默认配置
    assert "model_path" in config.config
    assert "conf_threshold" in config.config
    assert "names" in config.config
    logger.info("✓ 配置加载测试通过")
    
    # 测试配置保存
    test_config = {"test_key": "test_value"}
    config.save_config(test_config)
    
    # 验证文件是否创建
    assert os.path.exists(test_config_file)
    logger.info("✓ 配置保存测试通过")
    
    # 清理测试文件
    if os.path.exists(test_config_file):
        os.remove(test_config_file)

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

def main():
    """主测试函数"""
    logger.info("开始健壮性测试...")
    
    try:
        # 运行各个测试
        test_error_handler()
        test_resource_manager()
        test_thread_safe_counter()
        test_config()
        
        # 运行压力测试
        logger.info("\n" + "="*50)
        run_stress_test()
        
        logger.info("\n" + "="*50)
        logger.info("🎉 所有测试通过！代码健壮性良好。")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()