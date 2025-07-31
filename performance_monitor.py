import time
import statistics
import logging
from typing import Dict, List, Optional, Any
from collections import deque, defaultdict
import threading

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """性能监控类"""
    
    def __init__(self, max_samples: int = 100):
        self.max_samples = max_samples
        self.metrics = defaultdict(lambda: deque(maxlen=max_samples))
        self.current_session = {}
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def start_inference(self, session_id: str = "default") -> None:
        """开始推理计时"""
        with self.lock:
            self.current_session[session_id] = {
                'start_time': time.time(),
                'start_memory': self._get_memory_usage()
            }
    
    def end_inference(self, session_id: str = "default", 
                     provider: str = "unknown", 
                     model_name: str = "default") -> float:
        """结束推理计时并记录指标"""
        with self.lock:
            if session_id not in self.current_session:
                logger.warning(f"未找到会话 {session_id}")
                return 0.0
            
            session = self.current_session.pop(session_id)
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            inference_time = end_time - session['start_time']
            memory_usage = end_memory - session['start_memory']
            
            # 记录指标
            key = f"{provider}_{model_name}"
            self.metrics[f"{key}_inference_time"].append(inference_time)
            self.metrics[f"{key}_memory_usage"].append(memory_usage)
            self.metrics[f"{key}_fps"].append(1.0 / inference_time if inference_time > 0 else 0)
            
            return inference_time
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
        except Exception as e:
            logger.debug(f"获取内存使用量失败: {e}")
            return 0.0
    
    def get_stats(self, provider: str, model_name: str = "default") -> Dict[str, Any]:
        """获取指定提供者的统计信息"""
        with self.lock:
            key = f"{provider}_{model_name}"
            
            stats = {}
            
            # 推理时间统计
            if f"{key}_inference_time" in self.metrics:
                times = list(self.metrics[f"{key}_inference_time"])
                if times:
                    stats['inference_time'] = {
                        'avg': statistics.mean(times),
                        'min': min(times),
                        'max': max(times),
                        'std': statistics.stdev(times) if len(times) > 1 else 0,
                        'count': len(times)
                    }
            
            # FPS统计
            if f"{key}_fps" in self.metrics:
                fps_values = list(self.metrics[f"{key}_fps"])
                if fps_values:
                    stats['fps'] = {
                        'avg': statistics.mean(fps_values),
                        'min': min(fps_values),
                        'max': max(fps_values),
                        'std': statistics.stdev(fps_values) if len(fps_values) > 1 else 0,
                        'count': len(fps_values)
                    }
            
            # 内存使用统计
            if f"{key}_memory_usage" in self.metrics:
                memory_values = list(self.metrics[f"{key}_memory_usage"])
                if memory_values:
                    stats['memory_usage'] = {
                        'avg': statistics.mean(memory_values),
                        'min': min(memory_values),
                        'max': max(memory_values),
                        'std': statistics.stdev(memory_values) if len(memory_values) > 1 else 0,
                        'count': len(memory_values)
                    }
            
            return stats
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有提供者的统计信息"""
        with self.lock:
            all_stats = {}
            
            # 提取所有唯一的provider_model组合
            provider_models = set()
            for key in self.metrics.keys():
                if key.endswith('_inference_time'):
                    provider_model = key[:-15]  # 移除'_inference_time'后缀
                    provider_models.add(provider_model)
            
            for provider_model in provider_models:
                if '_' in provider_model:
                    provider, model = provider_model.rsplit('_', 1)
                else:
                    provider, model = provider_model, 'default'
                
                stats = self.get_stats(provider, model)
                if stats:
                    all_stats[provider_model] = stats
            
            return all_stats
    
    def get_comparison(self) -> Dict[str, Any]:
        """获取不同提供者的性能对比"""
        all_stats = self.get_all_stats()
        
        if not all_stats:
            return {}
        
        comparison = {
            'providers': list(all_stats.keys()),
            'best_fps': None,
            'best_latency': None,
            'lowest_memory': None,
            'rankings': {}
        }
        
        # 找出最佳性能
        best_fps = {'provider': None, 'value': 0}
        best_latency = {'provider': None, 'value': float('inf')}
        lowest_memory = {'provider': None, 'value': float('inf')}
        
        for provider, stats in all_stats.items():
            if 'fps' in stats and stats['fps']['avg'] > best_fps['value']:
                best_fps = {'provider': provider, 'value': stats['fps']['avg']}
            
            if 'inference_time' in stats and stats['inference_time']['avg'] < best_latency['value']:
                best_latency = {'provider': provider, 'value': stats['inference_time']['avg']}
            
            if 'memory_usage' in stats and stats['memory_usage']['avg'] < lowest_memory['value']:
                lowest_memory = {'provider': provider, 'value': stats['memory_usage']['avg']}
        
        comparison['best_fps'] = best_fps
        comparison['best_latency'] = best_latency
        comparison['lowest_memory'] = lowest_memory
        
        # 创建排名
        fps_ranking = sorted(all_stats.items(), 
                           key=lambda x: x[1].get('fps', {}).get('avg', 0), 
                           reverse=True)
        latency_ranking = sorted(all_stats.items(), 
                               key=lambda x: x[1].get('inference_time', {}).get('avg', float('inf')))
        memory_ranking = sorted(all_stats.items(), 
                              key=lambda x: x[1].get('memory_usage', {}).get('avg', float('inf')))
        
        comparison['rankings'] = {
            'fps': [provider for provider, _ in fps_ranking],
            'latency': [provider for provider, _ in latency_ranking],
            'memory': [provider for provider, _ in memory_ranking]
        }
        
        return comparison
    
    def get_recent_performance(self, provider: str, model_name: str = "default", 
                             samples: int = 10) -> Dict[str, float]:
        """获取最近的性能数据"""
        with self.lock:
            key = f"{provider}_{model_name}"
            recent = {}
            
            if f"{key}_inference_time" in self.metrics:
                times = list(self.metrics[f"{key}_inference_time"])[-samples:]
                if times:
                    recent['avg_inference_time'] = statistics.mean(times)
                    recent['recent_fps'] = 1.0 / recent['avg_inference_time']
            
            if f"{key}_memory_usage" in self.metrics:
                memory = list(self.metrics[f"{key}_memory_usage"])[-samples:]
                if memory:
                    recent['avg_memory_usage'] = statistics.mean(memory)
            
            return recent
    
    def reset_stats(self, provider: str = None, model_name: str = None) -> None:
        """重置统计信息"""
        with self.lock:
            if provider is None:
                # 重置所有统计
                self.metrics.clear()
                self.current_session.clear()
                self.start_time = time.time()
            else:
                # 重置特定提供者的统计
                key = f"{provider}_{model_name or 'default'}"
                keys_to_remove = [k for k in self.metrics.keys() if k.startswith(key)]
                for k in keys_to_remove:
                    del self.metrics[k]
    
    def export_stats(self) -> Dict[str, Any]:
        """导出所有统计数据"""
        with self.lock:
            export_data = {
                'timestamp': time.time(),
                'session_duration': time.time() - self.start_time,
                'stats': self.get_all_stats(),
                'comparison': self.get_comparison(),
                'raw_metrics': {k: list(v) for k, v in self.metrics.items()}
            }
            return export_data
    
    def log_summary(self) -> None:
        """记录性能摘要到日志"""
        comparison = self.get_comparison()
        
        if not comparison:
            logger.info("暂无性能数据")
            return
        
        logger.info("=== 性能监控摘要 ===")
        
        if comparison['best_fps']['provider']:
            logger.info(f"最高FPS: {comparison['best_fps']['provider']} "
                       f"({comparison['best_fps']['value']:.2f} fps)")
        
        if comparison['best_latency']['provider']:
            logger.info(f"最低延迟: {comparison['best_latency']['provider']} "
                       f"({comparison['best_latency']['value']*1000:.2f} ms)")
        
        if comparison['lowest_memory']['provider']:
            logger.info(f"最低内存: {comparison['lowest_memory']['provider']} "
                       f"({comparison['lowest_memory']['value']:.2f} MB)")
        
        logger.info(f"FPS排名: {' > '.join(comparison['rankings']['fps'][:3])}")
        logger.info(f"延迟排名: {' > '.join(comparison['rankings']['latency'][:3])}")


# 全局性能监控实例
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """获取性能监控实例（单例模式）"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor