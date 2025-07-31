# 游戏辅助程序 - 增强版

这是一个基于YOLO模型的游戏辅助程序，经过全面升级，支持多种显卡（AMD、NVIDIA、Intel）、多种推理模式，并提供现代化的GUI界面和性能监控功能。

## 🚀 新增功能亮点

### 1. 全面的GPU支持
- **AMD显卡**: 支持ROCm加速（Radeon RX系列、Instinct系列）
- **NVIDIA显卡**: 支持CUDA和TensorRT加速（GeForce、Quadro、Tesla系列）
- **Intel显卡**: 支持OpenVINO加速（核显、Arc系列）
- **自动检测**: 智能检测可用硬件并推荐最佳配置

### 2. 多种推理模式
- **自动模式**: 智能选择最佳推理后端
- **CPU模式**: 纯CPU推理，兼容性最佳
- **GPU模式**: 优先使用GPU加速
- **最佳性能模式**: 使用性能最优的后端
- **自定义模式**: 手动选择特定的执行提供者

### 3. 现代化GUI界面
- **直观配置**: 图形化配置模型、设备和参数
- **实时监控**: 显示FPS、延迟、内存使用等性能指标
- **性能对比**: 不同推理模式的性能图表对比
- **设备信息**: 详细的硬件检测和支持信息
- **日志管理**: 实时日志显示和导出功能

### 4. 智能性能优化
- **自动配置**: 根据硬件特性自动优化推理参数
- **性能监控**: 实时跟踪推理性能和资源使用
- **错误恢复**: 自动回退到兼容的执行提供者
- **内存管理**: 优化内存使用，防止内存泄漏

## 📋 系统要求

### 基础要求
- Python 3.8+
- 4GB+ RAM
- Windows 10/11, Ubuntu 20.04+, 或其他Linux发行版

### AMD GPU要求
- AMD Radeon RX 400系列或更新
- ROCm 5.0+ (Linux) 或 AMD GPU软件 (Windows)
- onnxruntime-rocm 包

### NVIDIA GPU要求
- NVIDIA GTX 1060 或更新的显卡
- CUDA 11.0+ 和兼容的驱动程序
- onnxruntime-gpu 包

### Intel GPU要求
- Intel Arc A系列 或 第11代Core处理器以上的核显
- OpenVINO 2023.0+
- openvino 包

## 🛠️ 安装指南

### 1. 基础安装
```bash
# 克隆项目
git clone <项目地址>
cd game-assistant-enhanced

# 安装基础依赖
pip install -r requirements.txt
```

### 2. GPU支持安装

#### AMD GPU (ROCm)
```bash
# Linux上安装ROCm (Ubuntu为例)
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dev

# 安装ROCm版本的ONNX Runtime
pip uninstall onnxruntime
pip install onnxruntime-rocm
```

#### NVIDIA GPU (CUDA)
```bash
# 安装CUDA Toolkit (参考NVIDIA官方文档)
# 然后安装GPU版本的ONNX Runtime
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

#### Intel GPU (OpenVINO)
```bash
# 安装OpenVINO
pip install openvino
```

### 3. 验证安装
```bash
# 检查环境和设备支持
python start_enhanced.py --check --verbose
```

## 🚀 使用方法

### 启动GUI界面（推荐）
```bash
python start_enhanced.py
```

### 启动控制台版本
```bash
python start_enhanced.py --console
```

### 环境检查
```bash
python start_enhanced.py --check
```

## 🎮 GUI界面使用指南

### 1. 配置设置
- **模型配置**: 选择或浏览ONNX模型文件
- **推理配置**: 选择推理模式和执行提供者
- **检测参数**: 调整置信度和NMS阈值

### 2. 设备信息
- 查看检测到的硬件设备
- 查看支持的执行提供者
- 检查ROCm/CUDA/OpenVINO支持状态

### 3. 性能监控
- 实时FPS、延迟、内存使用显示
- 不同提供者的性能图表对比
- 性能数据导出功能

### 4. 程序控制
- 启动/停止游戏辅助程序
- 保存/重置配置
- 日志查看和导出

## ⚙️ 推理模式详解

### 自动模式 (auto)
系统自动检测硬件并选择最佳执行提供者：
- 优先级: TensorRT > CUDA > ROCm > OpenVINO > CPU
- 自动配置优化参数
- 错误时自动回退

### CPU模式 (cpu)
使用CPU进行推理：
- 兼容性最佳，所有系统都支持
- 性能相对较低，但稳定可靠
- 适合低端硬件或调试使用

### GPU模式 (gpu)
优先使用GPU加速：
- 自动选择最佳GPU提供者
- 显著提升推理速度
- 需要对应的GPU驱动和运行库

### 最佳性能模式 (best_performance)
使用性能最优的配置：
- 根据性能测试选择最快的提供者
- 可能增加内存使用
- 推荐用于高性能需求

## 📊 性能优化建议

### AMD GPU优化
1. 确保安装最新的ROCm驱动
2. 设置适当的GPU内存限制
3. 启用HIP Graph优化
4. 监控GPU温度和功耗

### NVIDIA GPU优化
1. 使用CUDA 11.8+获得最佳兼容性
2. 启用TensorRT进行进一步加速
3. 调整CUDNN算法搜索策略
4. 使用FP16精度提升性能

### Intel GPU优化
1. 使用最新版OpenVINO
2. 启用GPU_FP16模式
3. 调整线程数配置
4. 考虑使用模型量化

### 通用优化
1. 选择合适的输入尺寸
2. 调整batch size（如果支持）
3. 启用内存池管理
4. 定期更新驱动程序

## 🔧 配置文件详解

主配置文件 `config.json` 包含以下主要部分：

### 模型配置
```json
{
    "model_path": "onnxmd/10w320v5.onnx",
    "conf_threshold": 0.4,
    "nms_threshold": 0.4,
    "input_size": [640, 640]
}
```

### 推理配置
```json
{
    "inference_mode": "auto",
    "providers": ["ROCMExecutionProvider"],
    "provider_options": [{
        "device_id": 0,
        "gpu_mem_limit": 2147483648,
        "enable_hip_graph": true
    }]
}
```

### 性能配置
```json
{
    "performance_monitoring": true,
    "auto_optimize": true
}
```

## 🐛 常见问题解决

### AMD GPU问题
```
错误: ROCMExecutionProvider 不可用
解决: 
1. 检查ROCm安装: rocm-smi
2. 安装onnxruntime-rocm: pip install onnxruntime-rocm
3. 检查GPU支持: cat /opt/rocm/bin/rocminfo
```

### NVIDIA GPU问题
```
错误: CUDAExecutionProvider 不可用
解决:
1. 检查CUDA安装: nvcc --version
2. 检查GPU状态: nvidia-smi
3. 安装GPU版本: pip install onnxruntime-gpu
```

### 性能问题
```
问题: FPS过低
解决:
1. 检查GPU使用率
2. 调整输入尺寸
3. 降低检测阈值
4. 使用TensorRT加速
```

### 内存问题
```
问题: 内存不足
解决:
1. 设置GPU内存限制
2. 减小batch size
3. 启用内存池
4. 定期清理缓存
```

## 📝 开发指南

### 添加新的执行提供者
1. 在 `device_manager.py` 中添加检测逻辑
2. 在 `get_provider_config()` 中添加配置
3. 更新GUI界面的提供者列表
4. 添加相应的依赖到 `requirements.txt`

### 自定义性能监控
1. 继承 `PerformanceMonitor` 类
2. 重写监控方法
3. 在GUI中集成新的监控指标
4. 更新图表显示逻辑

### 添加新的推理模式
1. 在 `Config` 类中添加模式处理
2. 更新GUI的模式选择器
3. 实现模式特定的优化
4. 更新文档说明

## 🔒 安全说明

1. **权限要求**: 程序需要屏幕捕获和鼠标控制权限
2. **防火墙**: 某些功能可能需要网络访问权限
3. **游戏规则**: 请遵守游戏规则，合理使用辅助功能
4. **数据隐私**: 程序不收集任何个人数据

## 📄 许可证

本项目仅供学习和研究使用，请遵守相关法律法规。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

### 贡献指南
1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 📞 支持

如果遇到问题，请：
1. 查看常见问题部分
2. 运行环境检查: `python start_enhanced.py --check --verbose`
3. 查看日志文件: `app.log`
4. 提交Issue并附上详细信息

---

**注意**: 使用任何游戏辅助工具都可能违反游戏服务条款，请谨慎使用并承担相应风险。