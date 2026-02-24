# IndexTTS2 - ROCm Deployment for AMD Strix Halo (gfx1151)

在 AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151) 上部署 IndexTTS2 模型，并且尽可能地优化速度。

一句话说清楚：请直接使用Dockerfile和docker-compose.yml来创建容器，它们会打各种补丁修复问题并让BIGVGAN在CPU上运行以绕过目前GPU上执行它过慢的问题。

## 快速启动

```bash
# 构建并启动
docker compose up -d

# 查看日志
docker compose logs -f

# WebUI
# http://localhost:8765
```

## 已知问题与修复

### 1. torchvision C++ 崩溃
**症状**: `terminate called: basic_string::_M_construct null not valid`
**原因**: torchvision 的 `_C.so` 在 gfx1151 上初始化时 C++ 层面 abort，无法用 try/except 捕获
**修复**: 不安装 torchvision。IndexTTS2 不直接需要它。

### 2. transformers 间接 import torchvision
**症状**: 即使卸载 torchvision，`from transformers import LlamaForCausalLM` 仍报错
**原因**: `transformers.loss.loss_utils` 无条件 import object detection loss 模块，它们依赖 torchvision
**修复**: Patch loss_utils.py，注释掉所有 object detection 相关的 import 和 LOSS_MAPPING 条目

### 3. torchaudio.save 崩溃
**症状**: `ImportError: TorchCodec is required for save_with_torchcodec`
**原因**: torchaudio 2.9.x 默认使用 torchcodec 后端，与 ROCm PyTorch 不兼容
**修复**: Patch `infer_v2.py`，用 `soundfile.write()` 替代 `torchaudio.save()`

### 4. CUDA vs ROCm 包冲突
**症状**: `OSError: libtorch_cuda.so: cannot open shared object file`
**原因**: pip 从 PyPI 安装 CUDA 版的 torch 生态包
**修复**: 
- torchaudio 从 ROCm 索引 (`download.pytorch.org/whl/rocm6.4`) 安装
- 使用 pip constraints 锁定版本，防止被覆盖

### 5. BigVGAN 性能 — 混合设备方案（已解决）
**症状**: 10秒语音 BigVGAN 阶段需要 ~82s（GPU）或 ~9s（CPU）
**原因**: gfx1151 的 MIOpen 卷积缺少优化内核，fallback 到慢速实现
**修复**: 混合设备方案 — BigVGAN 跑 CPU（Zen5 16核），GPT 和 S2Mel 留 GPU
**效果**:
| 模式 | GPT | S2Mel | BigVGAN | 总耗时 | RTF |
|------|-----|-------|---------|--------|-----|
| 纯 GPU fp32 | 8s | 8s | 82s | 113s | 10.5 |
| 混合 GPU+CPU fp32 | 8s | 8s | 1.3s | 18s | 2.4 |
| **混合 GPU+CPU fp16** | **7s** | **3s** | **1.3s** | **~12s** | **~2.3** |

### 6. numba Python 3.12 不兼容
**症状**: `numba==0.58.1 requires Python >=3.8,<3.12`
**修复**: Patch pyproject.toml，放宽到 `numba>=0.60.0`

## 文件说明

```
├── Dockerfile          # 主构建文件（含所有 ROCm patch）
├── docker-compose.yml  # 服务编排（GPU映射、持久化卷）
├── audio_prompts/      # 音色参考音频（映射到容器内）
├── outputs/            # 生成的语音输出
└── README.md           # 本文档
```

## 环境变量

| 变量 | 值 | 用途 |
|------|------|------|
| HSA_OVERRIDE_GFX_VERSION | 11.0.0 | gfx1151 → gfx1100 兼容模式 |
| HSA_ENABLE_SDMA | 0 | 禁用 SDMA，避免 Strix Halo 数据传输 bug |
| GPU_MAX_ALLOC_PERCENT | 100 | 允许使用全部显存 |
| TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL | 1 | 启用实验性 AOTriton 加速 |
| MIOPEN_LOG_LEVEL | 4 | MIOpen 日志（调试用，可设为 0） |
