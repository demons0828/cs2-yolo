#!/usr/bin/env python3
"""
游戏辅助程序 - 增强版启动脚本
支持自动设备检测、多种推理模式和GUI界面
"""

import os
import sys
import logging
import argparse
from typing import Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查必要的依赖"""
    missing_deps = []
    
    try:
        import tkinter
    except ImportError:
        missing_deps.append("tkinter")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import onnxruntime
    except ImportError:
        missing_deps.append("onnxruntime")
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import PIL
    except ImportError:
        missing_deps.append("Pillow")
    
    try:
        import pynput
    except ImportError:
        missing_deps.append("pynput")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    if missing_deps:
        logger.error(f"缺少必要依赖: {', '.join(missing_deps)}")
        logger.info("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_models():
    """检查模型文件"""
    model_dir = "onnxmd"
    if not os.path.exists(model_dir):
        logger.warning(f"模型目录不存在: {model_dir}")
        return False
    
    onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
    if not onnx_files:
        logger.warning(f"在 {model_dir} 目录中未找到ONNX模型文件")
        return False
    
    logger.info(f"找到 {len(onnx_files)} 个模型文件: {', '.join(onnx_files)}")
    return True

def setup_environment():
    """设置环境变量"""
    # 设置matplotlib后端为非交互式，避免可能的冲突
    os.environ['MPLBACKEND'] = 'TkAgg'
    
    # 如果是Windows，设置DPI感知
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass

def print_system_info():
    """打印系统信息"""
    logger.info("=== 系统信息 ===")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"操作系统: {os.name}")
    logger.info(f"工作目录: {os.getcwd()}")
    
    # 检查GPU支持
    try:
        from device_manager import get_device_manager
        device_manager = get_device_manager()
        summary = device_manager.get_device_summary()
        
        logger.info("=== 设备信息 ===")
        logger.info(f"推荐提供者: {summary['recommended_provider']}")
        logger.info(f"可用提供者: {', '.join(summary['available_providers'])}")
        
        for device_type, device_info in summary['devices'].items():
            if device_type == "CPU":
                logger.info(f"{device_type}: {device_info['cores']} 核心")
            else:
                logger.info(f"{device_type}: {device_info['count']} 个设备")
    except Exception as e:
        logger.warning(f"获取设备信息失败: {e}")

def start_gui():
    """启动GUI界面"""
    try:
        logger.info("启动增强版GUI界面...")
        from enhanced_gui import main as gui_main
        gui_main()
    except ImportError:
        logger.error("增强版GUI模块未找到，尝试启动原版GUI...")
        try:
            from gui import main as gui_main
            gui_main()
        except ImportError:
            logger.error("GUI模块未找到，请检查文件是否存在")
            return False
    except Exception as e:
        logger.error(f"启动GUI失败: {e}")
        return False
    
    return True

def start_console():
    """启动控制台版本"""
    try:
        logger.info("启动控制台版本...")
        from main import main as console_main
        console_main()
    except ImportError:
        logger.error("主程序模块未找到")
        return False
    except Exception as e:
        logger.error(f"启动控制台版本失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="游戏辅助程序 - 增强版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python start_enhanced.py              # 启动GUI界面
  python start_enhanced.py --console    # 启动控制台版本
  python start_enhanced.py --check      # 仅检查环境
        """
    )
    
    parser.add_argument(
        '--console', 
        action='store_true',
        help='启动控制台版本（无GUI）'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='仅检查环境和依赖'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细信息'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("游戏辅助程序 - 增强版")
    logger.info("=" * 50)
    
    # 设置环境
    setup_environment()
    
    # 打印系统信息
    if args.verbose or args.check:
        print_system_info()
    
    # 检查依赖
    if not check_dependencies():
        logger.error("依赖检查失败，程序退出")
        sys.exit(1)
    
    # 检查模型
    if not check_models():
        logger.warning("模型检查失败，程序可能无法正常工作")
        if args.check:
            sys.exit(1)
    
    if args.check:
        logger.info("环境检查完成")
        return
    
    # 启动程序
    if args.console:
        success = start_console()
    else:
        success = start_gui()
    
    if not success:
        logger.error("程序启动失败")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        sys.exit(1)