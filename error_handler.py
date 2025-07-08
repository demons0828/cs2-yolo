import logging
import traceback
import sys
import time
from typing import Callable, Any, Optional, Dict
from functools import wraps
import threading

logger = logging.getLogger(__name__)

class ErrorHandler:
    """错误处理工具类"""
    
    def __init__(self):
        self.error_count = 0
        self.max_retries = 3
        self.retry_delay = 1.0
        self.error_callbacks: Dict[str, Callable] = {}
    
    def handle_exception(self, exception: Exception, context: str = "") -> None:
        """
        处理异常并记录日志
        
        Args:
            exception: 异常对象
            context: 异常发生的上下文
        """
        self.error_count += 1
        error_msg = f"错误 #{self.error_count} - {context}: {str(exception)}"
        logger.error(error_msg)
        logger.debug(f"详细错误信息: {traceback.format_exc()}")
        
        # 调用错误回调函数
        if context in self.error_callbacks:
            try:
                self.error_callbacks[context](exception)
            except Exception as e:
                logger.error(f"错误回调函数执行失败: {e}")
    
    def retry_on_failure(self, max_retries: Optional[int] = None, delay: Optional[float] = None):
        """
        装饰器：在失败时重试
        
        Args:
            max_retries: 最大重试次数
            delay: 重试间隔（秒）
        """
        max_retries = max_retries if max_retries is not None else self.max_retries
        delay = delay if delay is not None else self.retry_delay
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries:
                            logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
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
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Optional[Any]:
        """
        安全执行函数，捕获所有异常
        
        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数
            
        Returns:
            函数返回值，如果失败则返回None
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_exception(e, f"执行函数 {func.__name__}")
            return None
    
    def register_error_callback(self, context: str, callback: Callable) -> None:
        """
        注册错误回调函数
        
        Args:
            context: 错误上下文
            callback: 回调函数
        """
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
    
    def register_resource(self, resource: Any, cleanup_func: Callable) -> None:
        """
        注册需要清理的资源
        
        Args:
            resource: 资源对象
            cleanup_func: 清理函数
        """
        self.resources.append((resource, cleanup_func))
    
    def register_cleanup_callback(self, callback: Callable) -> None:
        """
        注册清理回调函数
        
        Args:
            callback: 清理回调函数
        """
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

# 全局错误处理器实例
error_handler = ErrorHandler()
resource_manager = ResourceManager()

def handle_critical_error(exception: Exception) -> None:
    """处理严重错误"""
    logger.critical(f"严重错误: {exception}")
    logger.critical("程序将退出...")
    resource_manager.cleanup()
    sys.exit(1)

# 注册严重错误处理
error_handler.register_error_callback("critical", handle_critical_error)

def safe_thread(func: Callable) -> Callable:
    """
    装饰器：确保线程安全执行
    
    Args:
        func: 要执行的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_handler.handle_exception(e, f"线程执行 {func.__name__}")
            return None
    return wrapper

if __name__ == "__main__":
    # 测试代码
    @error_handler.retry_on_failure(max_retries=2, delay=0.1)
    def test_function(should_fail: bool = False):
        if should_fail:
            raise ValueError("测试错误")
        return "成功"
    
    # 测试重试机制
    print("测试重试机制...")
    try:
        result = test_function(should_fail=True)
        print(f"结果: {result}")
    except Exception as e:
        print(f"最终失败: {e}")
    
    # 测试安全执行
    print("\n测试安全执行...")
    result = error_handler.safe_execute(test_function, should_fail=True)
    print(f"安全执行结果: {result}")
    
    # 测试线程安全计数器
    print("\n测试线程安全计数器...")
    counter = ThreadSafeCounter()
    print(f"初始值: {counter.get_value()}")
    counter.increment(5)
    print(f"增加后: {counter.get_value()}")
    counter.decrement(2)
    print(f"减少后: {counter.get_value()}")