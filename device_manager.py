import os
import platform
import subprocess
import logging
from typing import List, Dict, Tuple, Optional
import onnxruntime as ort

logger = logging.getLogger(__name__)

class DeviceManager:
    """设备管理器，用于检测和管理推理设备"""
    
    def __init__(self):
        self.available_providers = []
        self.device_info = {}
        self.recommended_provider = None
        self._detect_devices()
    
    def _detect_devices(self) -> None:
        """检测可用的推理设备和提供者"""
        try:
            # 获取ONNX Runtime支持的所有提供者
            all_providers = ort.get_available_providers()
            logger.info(f"ONNX Runtime可用提供者: {all_providers}")
            
            # 检测各种设备
            self._detect_amd_devices()
            self._detect_nvidia_devices()
            self._detect_intel_devices()
            self._detect_cpu_info()
            
            # 根据检测结果确定可用提供者
            self._determine_available_providers(all_providers)
            
            # 推荐最佳提供者
            self._recommend_best_provider()
            
        except Exception as e:
            logger.error(f"设备检测失败: {e}")
            self.available_providers = ["CPUExecutionProvider"]
            self.recommended_provider = "CPUExecutionProvider"
    
    def _detect_amd_devices(self) -> None:
        """检测AMD显卡"""
        amd_info = {
            "available": False,
            "devices": [],
            "rocm_version": None,
            "hip_version": None
        }
        
        try:
            # 检查ROCm安装
            try:
                result = subprocess.run(['rocm-smi', '--showproductname'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    amd_info["available"] = True
                    # 解析GPU信息
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'GPU' in line and ':' in line:
                            gpu_name = line.split(':', 1)[1].strip()
                            amd_info["devices"].append(gpu_name)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.info("rocm-smi未找到，尝试其他方法检测AMD GPU")
            
            # 检查HIP版本
            try:
                result = subprocess.run(['hipconfig', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    amd_info["hip_version"] = result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # 在Linux上检查/sys/class/drm/
            if platform.system() == "Linux":
                try:
                    drm_cards = os.listdir('/sys/class/drm/')
                    for card in drm_cards:
                        if card.startswith('card') and not card.endswith('-'):
                            vendor_path = f'/sys/class/drm/{card}/device/vendor'
                            if os.path.exists(vendor_path):
                                with open(vendor_path, 'r') as f:
                                    vendor_id = f.read().strip()
                                    if vendor_id == '0x1002':  # AMD vendor ID
                                        amd_info["available"] = True
                                        # 尝试获取设备名称
                                        name_path = f'/sys/class/drm/{card}/device/product_name'
                                        if os.path.exists(name_path):
                                            with open(name_path, 'r') as f:
                                                device_name = f.read().strip()
                                                if device_name not in amd_info["devices"]:
                                                    amd_info["devices"].append(device_name)
                except Exception as e:
                    logger.debug(f"检查DRM设备失败: {e}")
            
            # Windows上检查WMI
            if platform.system() == "Windows":
                try:
                    import wmi
                    c = wmi.WMI()
                    for gpu in c.Win32_VideoController():
                        if gpu.Name and 'AMD' in gpu.Name.upper():
                            amd_info["available"] = True
                            if gpu.Name not in amd_info["devices"]:
                                amd_info["devices"].append(gpu.Name)
                except ImportError:
                    logger.debug("WMI模块未安装，无法检测Windows上的AMD GPU")
                except Exception as e:
                    logger.debug(f"WMI检测AMD GPU失败: {e}")
            
        except Exception as e:
            logger.error(f"AMD设备检测失败: {e}")
        
        self.device_info["amd"] = amd_info
        logger.info(f"AMD设备信息: {amd_info}")
    
    def _detect_nvidia_devices(self) -> None:
        """检测NVIDIA显卡"""
        nvidia_info = {
            "available": False,
            "devices": [],
            "cuda_version": None,
            "driver_version": None
        }
        
        try:
            # 使用nvidia-smi检测
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    nvidia_info["available"] = True
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            parts = line.split(',')
                            if len(parts) >= 2:
                                gpu_name = parts[0].strip()
                                driver_ver = parts[1].strip()
                                nvidia_info["devices"].append(gpu_name)
                                if not nvidia_info["driver_version"]:
                                    nvidia_info["driver_version"] = driver_ver
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.info("nvidia-smi未找到")
            
            # 检查CUDA版本
            try:
                result = subprocess.run(['nvcc', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    output = result.stdout
                    for line in output.split('\n'):
                        if 'release' in line.lower():
                            # 提取版本号
                            import re
                            match = re.search(r'release (\d+\.\d+)', line)
                            if match:
                                nvidia_info["cuda_version"] = match.group(1)
                                break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.debug("nvcc未找到")
            
            # 在Linux上检查/proc/driver/nvidia/version
            if platform.system() == "Linux" and os.path.exists('/proc/driver/nvidia/version'):
                try:
                    with open('/proc/driver/nvidia/version', 'r') as f:
                        content = f.read()
                        if 'NVIDIA' in content:
                            nvidia_info["available"] = True
                except Exception as e:
                    logger.debug(f"检查nvidia版本文件失败: {e}")
            
        except Exception as e:
            logger.error(f"NVIDIA设备检测失败: {e}")
        
        self.device_info["nvidia"] = nvidia_info
        logger.info(f"NVIDIA设备信息: {nvidia_info}")
    
    def _detect_intel_devices(self) -> None:
        """检测Intel显卡"""
        intel_info = {
            "available": False,
            "devices": [],
            "openvino_version": None
        }
        
        try:
            # 检查OpenVINO
            try:
                import openvino as ov
                intel_info["openvino_version"] = ov.__version__
                
                # 尝试创建Core来检测可用设备
                try:
                    core = ov.Core()
                    devices = core.available_devices
                    for device in devices:
                        if 'GPU' in device or 'INTEL' in device.upper():
                            intel_info["available"] = True
                            intel_info["devices"].append(device)
                except Exception as e:
                    logger.debug(f"OpenVINO设备检测失败: {e}")
                    
            except ImportError:
                logger.debug("OpenVINO未安装")
            
            # 在Linux上检查Intel GPU
            if platform.system() == "Linux":
                try:
                    drm_cards = os.listdir('/sys/class/drm/')
                    for card in drm_cards:
                        if card.startswith('card') and not card.endswith('-'):
                            vendor_path = f'/sys/class/drm/{card}/device/vendor'
                            if os.path.exists(vendor_path):
                                with open(vendor_path, 'r') as f:
                                    vendor_id = f.read().strip()
                                    if vendor_id == '0x8086':  # Intel vendor ID
                                        intel_info["available"] = True
                                        name_path = f'/sys/class/drm/{card}/device/product_name'
                                        if os.path.exists(name_path):
                                            with open(name_path, 'r') as f:
                                                device_name = f.read().strip()
                                                if device_name not in intel_info["devices"]:
                                                    intel_info["devices"].append(device_name)
                except Exception as e:
                    logger.debug(f"检查Intel DRM设备失败: {e}")
            
            # Windows上检查WMI
            if platform.system() == "Windows":
                try:
                    import wmi
                    c = wmi.WMI()
                    for gpu in c.Win32_VideoController():
                        if gpu.Name and 'INTEL' in gpu.Name.upper():
                            intel_info["available"] = True
                            if gpu.Name not in intel_info["devices"]:
                                intel_info["devices"].append(gpu.Name)
                except ImportError:
                    logger.debug("WMI模块未安装")
                except Exception as e:
                    logger.debug(f"WMI检测Intel GPU失败: {e}")
                    
        except Exception as e:
            logger.error(f"Intel设备检测失败: {e}")
        
        self.device_info["intel"] = intel_info
        logger.info(f"Intel设备信息: {intel_info}")
    
    def _detect_cpu_info(self) -> None:
        """检测CPU信息"""
        cpu_info = {
            "available": True,
            "cores": os.cpu_count(),
            "architecture": platform.machine(),
            "processor": platform.processor()
        }
        
        try:
            # 在Linux上获取更详细的CPU信息
            if platform.system() == "Linux" and os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        if line.startswith('model name'):
                            cpu_info["model"] = line.split(':', 1)[1].strip()
                            break
        except Exception as e:
            logger.debug(f"获取CPU详细信息失败: {e}")
        
        self.device_info["cpu"] = cpu_info
        logger.info(f"CPU信息: {cpu_info}")
    
    def _determine_available_providers(self, all_providers: List[str]) -> None:
        """根据设备检测结果确定可用的提供者"""
        available = []
        
        # AMD ROCm提供者
        if ("ROCMExecutionProvider" in all_providers and 
            self.device_info["amd"]["available"]):
            available.append("ROCMExecutionProvider")
        
        # NVIDIA CUDA提供者
        if ("CUDAExecutionProvider" in all_providers and 
            self.device_info["nvidia"]["available"]):
            available.append("CUDAExecutionProvider")
        
        # NVIDIA TensorRT提供者
        if ("TensorrtExecutionProvider" in all_providers and 
            self.device_info["nvidia"]["available"]):
            available.append("TensorrtExecutionProvider")
        
        # Intel OpenVINO提供者
        if ("OpenVINOExecutionProvider" in all_providers and 
            self.device_info["intel"]["available"]):
            available.append("OpenVINOExecutionProvider")
        
        # CPU提供者（始终可用）
        if "CPUExecutionProvider" in all_providers:
            available.append("CPUExecutionProvider")
        
        self.available_providers = available
        logger.info(f"可用执行提供者: {available}")
    
    def _recommend_best_provider(self) -> None:
        """推荐最佳执行提供者"""
        # 按性能优先级排序
        priority_order = [
            "TensorrtExecutionProvider",  # NVIDIA TensorRT (最快)
            "CUDAExecutionProvider",      # NVIDIA CUDA
            "ROCMExecutionProvider",      # AMD ROCm
            "OpenVINOExecutionProvider",  # Intel OpenVINO
            "CPUExecutionProvider"        # CPU (兜底选项)
        ]
        
        for provider in priority_order:
            if provider in self.available_providers:
                self.recommended_provider = provider
                break
        
        logger.info(f"推荐的执行提供者: {self.recommended_provider}")
    
    def get_provider_config(self, provider: str) -> Dict:
        """获取特定提供者的配置"""
        configs = {
            "ROCMExecutionProvider": {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                "enable_hip_graph": True
            },
            "CUDAExecutionProvider": {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                "cudnn_conv_algo_search": "EXHAUSTIVE"
            },
            "TensorrtExecutionProvider": {
                "device_id": 0,
                "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2GB
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True
            },
            "OpenVINOExecutionProvider": {
                "device_type": "GPU_FP16" if self.device_info["intel"]["available"] else "CPU",
                "precision": "FP16",
                "num_of_threads": self.device_info["cpu"]["cores"]
            },
            "CPUExecutionProvider": {
                "intra_op_num_threads": self.device_info["cpu"]["cores"],
                "inter_op_num_threads": 1
            }
        }
        
        return configs.get(provider, {})
    
    def get_device_summary(self) -> Dict:
        """获取设备检测摘要"""
        summary = {
            "available_providers": self.available_providers,
            "recommended_provider": self.recommended_provider,
            "devices": {}
        }
        
        # AMD设备摘要
        if self.device_info["amd"]["available"]:
            summary["devices"]["AMD"] = {
                "count": len(self.device_info["amd"]["devices"]),
                "devices": self.device_info["amd"]["devices"],
                "rocm_support": self.device_info["amd"]["hip_version"] is not None
            }
        
        # NVIDIA设备摘要
        if self.device_info["nvidia"]["available"]:
            summary["devices"]["NVIDIA"] = {
                "count": len(self.device_info["nvidia"]["devices"]),
                "devices": self.device_info["nvidia"]["devices"],
                "cuda_version": self.device_info["nvidia"]["cuda_version"],
                "driver_version": self.device_info["nvidia"]["driver_version"]
            }
        
        # Intel设备摘要
        if self.device_info["intel"]["available"]:
            summary["devices"]["Intel"] = {
                "count": len(self.device_info["intel"]["devices"]),
                "devices": self.device_info["intel"]["devices"],
                "openvino_version": self.device_info["intel"]["openvino_version"]
            }
        
        # CPU摘要
        summary["devices"]["CPU"] = {
            "cores": self.device_info["cpu"]["cores"],
            "architecture": self.device_info["cpu"]["architecture"]
        }
        
        return summary
    
    def validate_provider(self, provider: str) -> Tuple[bool, str]:
        """验证提供者是否可用"""
        if provider not in self.available_providers:
            return False, f"提供者 {provider} 不可用"
        
        try:
            # 尝试创建一个简单的推理会话来验证
            # 这里我们创建一个最小的测试模型
            import numpy as np
            
            # 创建一个简单的恒等模型用于测试
            model_proto = None
            try:
                import onnx
                from onnx import helper, TensorProto
                
                # 创建一个简单的恒等操作模型
                X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 224, 224])
                Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 224, 224])
                identity_node = helper.make_node('Identity', ['X'], ['Y'])
                graph = helper.make_graph([identity_node], 'test_graph', [X], [Y])
                model_proto = helper.make_model(graph)
                
                # 序列化模型
                model_bytes = model_proto.SerializeToString()
                
                # 尝试创建推理会话
                providers = [provider] if provider != "CPUExecutionProvider" else ["CPUExecutionProvider"]
                provider_options = [self.get_provider_config(provider)] if provider != "CPUExecutionProvider" else [{}]
                
                session = ort.InferenceSession(model_bytes, providers=providers, provider_options=provider_options)
                
                # 进行一次简单的推理测试
                input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
                output = session.run(None, {'X': input_data})
                
                return True, f"提供者 {provider} 验证成功"
                
            except ImportError:
                # 如果没有onnx库，跳过详细验证
                return True, f"提供者 {provider} 可用（跳过详细验证）"
                
        except Exception as e:
            return False, f"提供者 {provider} 验证失败: {str(e)}"


def get_device_manager() -> DeviceManager:
    """获取设备管理器实例（单例模式）"""
    if not hasattr(get_device_manager, '_instance'):
        get_device_manager._instance = DeviceManager()
    return get_device_manager._instance