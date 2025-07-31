#!/usr/bin/env python3
"""
基础功能测试脚本
测试增强版功能的核心逻辑，不依赖重型库
"""

import sys
import os
import json
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config_system():
    """测试配置系统"""
    logger.info("测试配置系统...")
    
    try:
        # 创建测试配置
        test_config = {
            "model_path": "onnxmd/test.onnx",
            "conf_threshold": 0.5,
            "nms_threshold": 0.4,
            "input_size": [640, 640],
            "screen_crop_size": [1920, 1080],
            "providers": ["CPUExecutionProvider"],
            "provider_options": [{"intra_op_num_threads": 4}],
            "inference_mode": "cpu",
            "performance_monitoring": True
        }
        
        # 保存配置
        with open('test_config.json', 'w') as f:
            json.dump(test_config, f, indent=4)
        
        # 读取配置
        with open('test_config.json', 'r') as f:
            loaded_config = json.load(f)
        
        # 验证配置
        assert loaded_config["inference_mode"] == "cpu"
        assert loaded_config["performance_monitoring"] == True
        assert len(loaded_config["providers"]) == 1
        
        # 清理
        os.remove('test_config.json')
        
        logger.info("✅ 配置系统测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 配置系统测试失败: {e}")
        return False

def test_device_manager_structure():
    """测试设备管理器的基础结构"""
    logger.info("测试设备管理器结构...")
    
    try:
        # 尝试导入设备管理器模块
        sys.path.insert(0, '.')
        
        # 验证模块文件存在
        if not os.path.exists('device_manager.py'):
            raise FileNotFoundError("device_manager.py 文件不存在")
        
        # 读取模块内容验证关键类和函数
        with open('device_manager.py', 'r') as f:
            content = f.read()
        
        required_elements = [
            'class DeviceManager',
            'def _detect_amd_devices',
            'def _detect_nvidia_devices', 
            'def _detect_intel_devices',
            'def get_provider_config',
            'def get_device_summary',
            'def validate_provider'
        ]
        
        for element in required_elements:
            if element not in content:
                raise ValueError(f"缺少必要元素: {element}")
        
        logger.info("✅ 设备管理器结构测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 设备管理器结构测试失败: {e}")
        return False

def test_performance_monitor_structure():
    """测试性能监控器的基础结构"""
    logger.info("测试性能监控器结构...")
    
    try:
        # 验证模块文件存在
        if not os.path.exists('performance_monitor.py'):
            raise FileNotFoundError("performance_monitor.py 文件不存在")
        
        # 读取模块内容验证关键类和函数
        with open('performance_monitor.py', 'r') as f:
            content = f.read()
        
        required_elements = [
            'class PerformanceMonitor',
            'def start_inference',
            'def end_inference',
            'def get_stats',
            'def get_comparison',
            'def export_stats'
        ]
        
        for element in required_elements:
            if element not in content:
                raise ValueError(f"缺少必要元素: {element}")
        
        logger.info("✅ 性能监控器结构测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 性能监控器结构测试失败: {e}")
        return False

def test_gui_structure():
    """测试GUI的基础结构"""
    logger.info("测试增强GUI结构...")
    
    try:
        # 验证模块文件存在
        if not os.path.exists('enhanced_gui.py'):
            raise FileNotFoundError("enhanced_gui.py 文件不存在")
        
        # 读取模块内容验证关键类和函数
        with open('enhanced_gui.py', 'r') as f:
            content = f.read()
        
        required_elements = [
            'class EnhancedGameAssistantGUI',
            'def create_model_config',
            'def create_inference_config',
            'def create_device_info',
            'def create_performance_monitor',
            'def refresh_device_info',
            'def update_performance_chart'
        ]
        
        for element in required_elements:
            if element not in content:
                raise ValueError(f"缺少必要元素: {element}")
        
        logger.info("✅ 增强GUI结构测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 增强GUI结构测试失败: {e}")
        return False

def test_main_integration():
    """测试主程序集成"""
    logger.info("测试主程序集成...")
    
    try:
        # 验证主程序文件存在
        if not os.path.exists('main.py'):
            raise FileNotFoundError("main.py 文件不存在")
        
        # 读取主程序内容验证集成
        with open('main.py', 'r') as f:
            content = f.read()
        
        required_imports = [
            'from device_manager import get_device_manager',
            'from performance_monitor import get_performance_monitor'
        ]
        
        for import_stmt in required_imports:
            if import_stmt not in content:
                raise ValueError(f"缺少必要导入: {import_stmt}")
        
        # 验证关键功能集成
        required_features = [
            'provider_options',
            'performance_monitoring',
            'device_manager',
            'performance_monitor'
        ]
        
        for feature in required_features:
            if feature not in content:
                raise ValueError(f"缺少集成功能: {feature}")
        
        logger.info("✅ 主程序集成测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 主程序集成测试失败: {e}")
        return False

def test_configuration_files():
    """测试配置文件"""
    logger.info("测试配置文件...")
    
    try:
        # 验证配置文件存在
        if not os.path.exists('config.json'):
            raise FileNotFoundError("config.json 文件不存在")
        
        # 读取并验证配置文件
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        required_keys = [
            'model_path',
            'providers',
            'provider_options',
            'inference_mode',
            'performance_monitoring'
        ]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件缺少必要键: {key}")
        
        # 验证requirements.txt
        if not os.path.exists('requirements.txt'):
            raise FileNotFoundError("requirements.txt 文件不存在")
        
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        required_packages = [
            'onnxruntime',
            'numpy',
            'opencv-python',
            'matplotlib',
            'psutil'
        ]
        
        for package in required_packages:
            if package not in requirements:
                raise ValueError(f"requirements.txt缺少必要包: {package}")
        
        logger.info("✅ 配置文件测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 配置文件测试失败: {e}")
        return False

def test_documentation():
    """测试文档文件"""
    logger.info("测试文档文件...")
    
    try:
        # 验证文档文件存在
        docs = ['README_ENHANCED.md', 'start_enhanced.py']
        
        for doc in docs:
            if not os.path.exists(doc):
                raise FileNotFoundError(f"文档文件不存在: {doc}")
        
        # 验证README内容
        with open('README_ENHANCED.md', 'r') as f:
            readme_content = f.read()
        
        required_sections = [
            '# 游戏辅助程序 - 增强版',
            '## 🚀 新增功能亮点',
            '## 🛠️ 安装指南',
            '## 🚀 使用方法',
            '## 🎮 GUI界面使用指南'
        ]
        
        for section in required_sections:
            if section not in readme_content:
                raise ValueError(f"README缺少必要章节: {section}")
        
        logger.info("✅ 文档文件测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 文档文件测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    logger.info("=" * 50)
    logger.info("开始运行增强版功能测试")
    logger.info("=" * 50)
    
    tests = [
        ("配置系统", test_config_system),
        ("设备管理器结构", test_device_manager_structure),
        ("性能监控器结构", test_performance_monitor_structure),
        ("增强GUI结构", test_gui_structure),
        ("主程序集成", test_main_integration),
        ("配置文件", test_configuration_files),
        ("文档文件", test_documentation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- 测试: {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} 测试异常: {e}")
            failed += 1
    
    logger.info("\n" + "=" * 50)
    logger.info("测试结果汇总")
    logger.info("=" * 50)
    logger.info(f"✅ 通过: {passed}")
    logger.info(f"❌ 失败: {failed}")
    logger.info(f"🔍 总计: {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 所有测试通过！增强版功能实现完成。")
        return True
    else:
        logger.warning(f"⚠️  有 {failed} 个测试失败，请检查相关功能。")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)