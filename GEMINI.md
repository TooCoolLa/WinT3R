# WinT3R 项目指南

WinT3R (Window-Based Streaming Reconstruction with Camera Token Pool) 是一个最先进的联机（Online）3D 重建模型，能够从图像流中实时推断精确的相机姿态和高质量的点云图。

## 核心技术栈
- **语言**: Python 3.10
- **深度学习框架**: PyTorch 2.5.1
- **核心库**: torchvision, torchaudio, transformers, opencv-python, plyfile
- **基础架构**: 基于 CroCo, DUSt3R, MASt3R, CUT3R 等项目改进

## 快速开始

### 环境配置
建议使用 Conda 环境：
```bash
conda create -n WinT3R python=3.10
conda activate WinT3R
# 安装 PyTorch (请根据 CUDA 版本调整)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
# 安装依赖
pip install -r requirements.txt
```

### 模型权重
从 [HuggingFace](https://huggingface.co/lizizun/WinT3R/resolve/main/pytorch_model.bin) 下载 `pytorch_model.bin` 并放置在 `checkpoints/` 目录下。

### 运行推理
```bash
# 使用示例图像进行默认推理
python recon.py

# 使用自定义数据进行在线模式推理
python recon.py --data_path <路径> --inference_mode online
```

## 项目结构与架构

### 关键目录与文件
- `recon.py`: 推理入口点，处理图像/视频加载、模型运行及点云导出。
- `dust3r/wint3r.py`: 核心模型类 `WinT3R`，实现了窗口化推理逻辑。
- `dust3r/blocks.py`: 定义了全局-局部解码器块（GlobalLocalDecoderBlock）等关键组件。
- `layers/`: 包含相机头（CameraHead）、深度头（PtsHead）、几何变换和姿态编码等模块。
- `croco/`: 包含 CroCo 主干网络相关的代码。

### 核心设计概念
1. **在线窗口机制 (Online Window Mechanism)**: 允许在同一窗口内以及跨相邻窗口进行图像令牌的充分交互。
2. **相机令牌池 (Camera Token Pool)**: 作为轻量级的“全局存储器”，从全局视角提升相机姿态预测质量。
3. **推理模式**:
   - `online`: 逐窗口实时编码，节省内存，适用于流式传输。
   - `offline`: 预先编码所有视图，再进行窗口处理，质量更高。

## 开发与协作规范

### 编码规范
- **四元数**: 全局使用 **XYZW** (scalar-last) 顺序。
- **姿态编码**: 使用 `absT_quaR` (7维：3位移 + 4四元数)。
- **图像处理**:
  - 默认分辨率为 512x384。
  - 必须为 16 的倍数（patch_size=16）。
  - 使用 ImageNet 的均值/标准差进行归一化。
- **设备**: 默认优先使用 `cuda`，若不可用则回退至 `cpu`。

### 验证与测试
- 修改核心模型后，应运行 `recon.py` 验证重建效果。
- 检查输出的 `.ply` 文件（默认在 `output/`）以确保几何一致性。

### 注意事项
- `croco/` 目录下的代码遵循其原有的 CC BY-NC-SA 4.0 许可。
- 训练和评估代码目前尚未完全开源（参见 TODO）。
