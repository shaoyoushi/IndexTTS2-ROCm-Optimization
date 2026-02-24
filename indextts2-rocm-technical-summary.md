# IndexTTS2 ROCm 部署技术总结

> 硬件：AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151)，128GB RAM / 96GB VRAM，Ubuntu 24.04
> 日期：2025-02-23 ~ 2025-02-24，和Claude一起鏖战了几个小时的结果。
> 最终性能：~12s 生成 10s 音频，RTF ≈ 2.3（从初始 RTF 10.5 优化 9.4 倍）
---

## 一、架构概览

IndexTTS2 推理流程由三个模型串行组成：

```
文本 → [GPT] → codes → [S2Mel] → mel → [BigVGAN] → wav
         自回归生成      CFM扩散(25步)      声码器卷积
```

最终部署架构：混合设备 + 混合精度

| 模块 | 设备 | 精度 | 耗时 | 占比 |
|------|------|------|------|------|
| GPT | GPU | FP16 | ~7s | 55% |
| S2Mel | GPU | FP16 | ~3s | 25% |
| BigVGAN | CPU | FP32 | ~1.3s | 10% |
| 其他(加载/IO) | - | - | ~1s | 10% |
| **总计** | | | **~12s** | |

---

## 二、gfx1151 ROCm 兼容性问题与解决

### 2.1 架构映射：gfx1151 → gfx1100

Strix Halo 的 GPU 架构标识为 `gfx1151`，不在 AMD ROCm 官方支持矩阵中。通过环境变量伪装为 `gfx1100`（Navi 31, RX 7900 系列）使 ROCm 运行时接受此 GPU：

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

两者同属 RDNA3 架构，指令集大部分兼容，但 gfx1151 有细微 ISA 差异（如 `row_bcast:15` 等 cross-lane 操作的编码不同），导致某些预编译内核不可用。

### 2.2 CWSR 内存页面错误（内核级）

**症状**：GPU 计算过程中间歇性出现 `memory access fault`，有时导致整个 ROCm 运行时失去对 GPU 的感知。

**根因**：gfx1151 的 CWSR（Compute Wave Save/Restore）功能存在 bug，在 GPU 计算任务上下文切换时触发页面错误。参见 ROCm/ROCm#5616。

**解决**：宿主机 GRUB 内核启动参数添加：
```
amdgpu.cwsr_enable=0
```
具体操作：
```bash
sudo nano /etc/default/grub
# GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amdgpu.cwsr_enable=0"
sudo update-grub && sudo reboot
```

**注意**：这是内核级参数，必须在宿主机修改，Docker 容器共享宿主机内核。CWSR 禁用的副作用是 GPU 失去计算任务抢占能力，但对单用户推理场景几乎无影响。

### 2.3 SDMA 数据传输 Bug

```bash
export HSA_ENABLE_SDMA=0
```

禁用 SDMA（System DMA）引擎，避免 Strix Halo 上已知的数据传输 bug。数据搬运改由着色器引擎完成，略有额外开销但保证正确性。

### 2.4 MIOpen 卷积内核缺失（核心性能瓶颈）

**这是 gfx1151 上最严重的性能问题。**

MIOpen 有 40+ 种卷积求解器（solver），每个都有 `IsApplicable()` 检查。gfx1151 不在任何求解器的验证矩阵中，导致绝大多数求解器返回 `false`，最终 fallback 到最慢的通用实现。

**表现**：BigVGAN vocoder（纯卷积）在 GPU 上跑一个 10s 片段需要 82s（相比 NVIDIA GPU 的 0.14s）。

**为什么不能自己修**：
- 不是缺代码，而是缺测试/验证。每个 solver 需要在目标硬件上验证正确性。
- 部分 solver 使用手写汇编（ISA），gfx1151 与 gfx1100 有细微指令编码差异。
- MIOpen 的 Kernel Database（KDB）包含 5 万+ 预编译内核，生成 gfx1151 版需要数天的穷举搜索。
- 构建依赖链复杂：ROCm compiler → Composable Kernel → rocBLAS → MIOpen。

**我们的解决方案**：将 BigVGAN 整个模型搬到 CPU 运行（见第四节）。

### 2.5 Flash Attention 不可用

Flash Attention 2 是 CUDA-only 的，在 ROCm 上不可用。ROCm 的替代方案是 SDPA（Scaled Dot-Product Attention），通过 AOTriton 后端实现：

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

IndexTTS2 内部使用 SDPA 作为 attention 实现，在 gfx1151 上可正常工作。

---

## 三、依赖兼容性问题与解决

### 3.1 PyTorch ROCm 版本锁定

**问题**：`pip install ".[webui]"` 可能从 PyPI 拉取 CUDA 版本的 torch/torchaudio，覆盖容器中的 ROCm 版本，导致 `libtorch_cuda.so: cannot open shared object file`。

**解决**：pip constraints 机制锁定版本：
```bash
# 先从 ROCm 索引安装 torchaudio
pip install torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# 生成 constraints 文件
python -c "from importlib.metadata import version; ..."
# 输出：torch==2.9.1  torchaudio==2.9.1

# 在 constraints 保护下安装项目依赖
pip install -c constraints.txt ".[webui]"
```

**关键细节**：constraints 生成时使用 `importlib.metadata` 而不是 `import torch`，因为 Docker build 阶段没有 GPU，ROCm 的 C++ 扩展初始化会失败。

### 3.2 torchvision C++ 崩溃

**症状**：`terminate called: basic_string::_M_construct null not valid` — C++ 层面直接 abort 进程，Python 的 try/except 无法捕获。

**根因**：torchvision 的 `_C.so` 在 import 时会查询 GPU 设备属性，gfx1151 的某个属性返回 null 指针，传给 `std::string` 构造函数导致 abort。

**解决**（分两步）：
1. **卸载 torchvision**：IndexTTS2 不直接使用它。
2. **Patch transformers**：`transformers.loss.loss_utils` 在模块级别无条件 import 了 object detection loss 模块（DFine、DeformableDetr、GroundingDino 等），这些模块依赖 torchvision。注释掉这些 import 和对应的 `LOSS_MAPPING` 条目。

**教训**：C++ 级别的 abort 不受 Python 异常机制保护，唯一解法是彻底阻止 import 链。

### 3.3 torchaudio.save 不兼容

**症状**：`ImportError: TorchCodec is required for save_with_torchcodec`

**根因**：torchaudio 2.9.x 默认使用 torchcodec 后端保存音频，torchcodec 与 ROCm PyTorch 不兼容。

**解决**：Patch `infer_v2.py`，用 `soundfile.write()` 替代 `torchaudio.save()`：
```python
# 原代码
torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
# 替换为
import soundfile as sf
sf.write(output_path, wav.type(torch.int16).squeeze().cpu().numpy(), sampling_rate)
```

### 3.4 numba Python 3.12 不兼容

**问题**：IndexTTS2 的 `pyproject.toml` 锁定 `numba==0.58.1`，该版本不支持 Python 3.12（ROCm 基础镜像使用 3.12）。

**解决**：放宽版本约束为 `numba>=0.60.0`。

---

## 四、性能优化历程

### 4.1 初始状态（纯 GPU FP32）

| 模块 | 耗时 | 说明 |
|------|------|------|
| GPT | 12s | 自回归生成，受内存带宽限制 |
| S2Mel | 10s | 25步 CFM diffusion |
| BigVGAN | 82s | MIOpen 卷积 fallback，极慢 |
| **总计** | **113s** | **RTF ≈ 10.5** |

### 4.2 BigVGAN → CPU（9x 加速）

**发现**：BigVGAN 的卷积在 GPU 上因 MIOpen fallback 极慢（82s），但 CPU（Zen5 16核）只需 ~1.3s。CPU 上使用的是 PyTorch 原生的 C++ 卷积实现（调用 MKL/OpenBLAS），完全绕过了 MIOpen。

**实现**：
- `webui.py`：模型加载后执行 `tts.bigvgan = tts.bigvgan.to("cpu").float()`
- `infer_v2.py`：BigVGAN forward call 前加 `.cpu()` 转换输入 tensor

**关键细节**：BigVGAN 只有一个调用点（`infer_v2.py` 第 659 行），且是推理流程的最后一步，输入输出边界清晰，非常适合单独搬到不同设备。

| 模块 | 优化前 | 优化后 |
|------|--------|--------|
| BigVGAN | 82s (GPU) | 1.3s (CPU) |
| 总计 | 113s | ~18s |

### 4.3 FP16 混合精度（S2Mel 2.4x 加速）

启用 `use_fp16=True` 后：

| 模块 | FP32 | FP16 | 加速比 | 说明 |
|------|------|------|--------|------|
| GPT | 7.79s | 6.82s | 1.14x | 内存带宽瓶颈，FP16 减半带宽需求 |
| S2Mel | 7.88s | 3.33s | 2.4x | 计算密集型，RDNA3 FP16 吞吐 2x |
| BigVGAN | 1.3s | 1.3s | - | 已在 CPU，保持 FP32 |

**为什么 GPT 提升小但 S2Mel 提升大**：
- GPT 是自回归生成（逐 token），每步读取整个 KV cache 但计算很少 → **内存带宽瓶颈**。FP16 减半数据量，但带宽天花板（212GB/s）不变，只提升 14%。
- S2Mel 是 25 步 CFM diffusion，每步是大矩阵运算 → **计算密集型**。FP16 让 RDNA3 的 FP16 ALU 吞吐翻倍，效果显著。

**BigVGAN 在 FP16 模式下的处理**：`use_fp16=True` 会把所有模型转 FP16，但 BigVGAN 搬到 CPU 后必须转回 FP32（`.float()`），因为 CPU 跑 FP16 反而更慢（x86 CPU 对 FP16 计算没有硬件加速，需要 FP32↔FP16 转换开销）。

### 4.4 S2Mel → CPU 的尝试（失败）

我们也尝试了将 S2Mel 搬到 CPU，但遇到两个障碍：

1. **调用点复杂**：S2Mel 有 4 个调用点，涉及 length_regulator、gpt_layer、cfm 三个子模块，内部 tensor 设备交错（mask、freqs_cis 等预计算 buffer 分散在各处），patch 成本极高。
2. **CPU 性能不佳**：S2Mel 在 CPU 上需要 25.77s（GPU FP16 只要 3.33s），因为它是 25 步 diffusion transformer，大量 attention + FFN，CPU 完全无优势。

**结论**：S2Mel 的计算特性（矩阵乘法密集）适合 GPU，与 BigVGAN 的卷积密集型完全不同。

### 4.5 FP8 可行性分析（不可行）

RDNA3 有部分 FP8 硬件单元，但：
- PyTorch + ROCm 对消费级 GPU 的 FP8 推理支持为零
- FP8 仅在 MI300X 数据中心 GPU 上成熟
- IndexTTS2 没有 FP8 代码路径

### 4.6 优化全历程汇总

| 阶段 | GPT | S2Mel | BigVGAN | 总计 | RTF | 累计加速 |
|------|-----|-------|---------|------|-----|----------|
| 初始 (GPU FP32) | 12s | 10s | 82s | 113s | 10.5 | 1x |
| +BigVGAN→CPU | 5s | 5s | 1.3s | 15s | 2.6 | 7.5x |
| +FP16 | 7s | 3s | 1.3s | ~12s | 2.3 | **9.4x** |

---

## 五、Docker 部署配置

### 5.1 基础镜像

```
rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1
```

选择此镜像的原因：
- 预装 ROCm 6.4 + PyTorch + conda 环境
- Ubuntu 24.04 匹配宿主机
- Python 3.12 是当前 ROCm 镜像的标配

### 5.2 必需的环境变量

| 变量 | 值 | 用途 |
|------|------|------|
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | gfx1151 → gfx1100 架构映射 |
| `GPU_MAX_ALLOC_PERCENT` | `100` | 允许分配全部显存 |
| `GPU_MAX_HEAP_SIZE` | `100` | 堆内存不限制 |
| `HSA_ENABLE_SDMA` | `0` | 禁用 SDMA 避免数据传输 bug |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1` | 启用实验性 AOTriton 加速（SDPA attention） |
| `MIOPEN_USER_DB_PATH` | `/app/miopen_cache` | MIOpen 内核缓存持久化路径 |
| `MIOPEN_LOG_LEVEL` | `4` | MIOpen 日志级别（调试用，生产可设为 0） |

### 5.3 Docker 设备映射

```yaml
devices:
  - /dev/kfd:/dev/kfd     # ROCm 内核融合驱动
  - /dev/dri:/dev/dri     # GPU 渲染设备
group_add:
  - video                  # GPU 访问权限
  - "992"                  # render group GID（容器内可能无此组名）
security_opt:
  - seccomp=unconfined     # ROCm 需要解除 seccomp 限制
cap_add:
  - SYS_PTRACE             # ROCm 调试/profiling 需要
```

### 5.4 持久化 Volume

| Volume | 容器路径 | 说明 |
|--------|----------|------|
| `indextts2_checkpoints` | `/app/checkpoints` | 模型权重（~3GB），避免重复下载 |
| `indextts2_miopen_cache` | `/app/miopen_cache` | MIOpen 内核编译缓存 |
| `./audio_prompts` | `/app/audio_prompts` | 音色参考音频（bind mount） |
| `./outputs` | `/app/outputs` | 生成的语音输出（bind mount） |

### 5.5 镜像管理

当前使用 `docker commit` 保存了包含所有手工优化的镜像：
```bash
docker commit indextts2 indextts2-rocm:optimized
```

备份命令：
```bash
docker save indextts2-rocm:optimized | gzip > ~/indextts2-rocm-optimized.tar.gz
# 恢复：docker load < ~/indextts2-rocm-optimized.tar.gz
```

**注意**：`docker commit` 不包含 named volume 的内容。模型权重存储在 `indextts2_checkpoints` volume 中，compose 启动时自动挂载，不需要镜像内包含。

---

## 六、Dockerfile 中的所有 Patch 清单

| 序号 | 修改对象 | 修改内容 | 原因 |
|------|----------|----------|------|
| 1 | `pyproject.toml` | torch 版本从 `==2.8.*` 放宽为 `>=2.7.0` | ROCm 基础镜像 torch 版本不匹配 |
| 2 | `pyproject.toml` | torchaudio 同上 | 同上 |
| 3 | `pyproject.toml` | numba 从 `==0.58.1` 放宽为 `>=0.60.0` | 0.58.1 不支持 Python 3.12 |
| 4 | pip | 卸载 torchvision | `_C.so` 在 gfx1151 上 C++ abort |
| 5 | `transformers/loss/loss_utils.py` | 注释 5 行 object detection import + 删除 10 条 LOSS_MAPPING | 阻断 torchvision import 链 |
| 6 | `indextts/infer_v2.py` | `torchaudio.save()` → `soundfile.write()` | torchaudio 2.9 torchcodec 不兼容 ROCm |
| 7 | `indextts/infer_v2.py` | BigVGAN 调用前加 `.cpu()` | 混合设备模式 |
| 8 | `webui.py` | 模型初始化后 `tts.bigvgan.to("cpu").float()` | BigVGAN 搬到 CPU 并保持 FP32 |

---

## 七、当前瓶颈与未来优化方向

### 当前瓶颈

GPT 自回归生成（~7s，占 55%）是最大瓶颈。这是内存带宽限制的本质问题：
- gfx1151 共享内存带宽 ~212GB/s
- 每生成一个 token 需要读取整个 KV cache
- ~269 tokens / 7s ≈ 38 token/s

### 理论可行但工程量大的方向

| 方向 | 原理 | 预期收益 | 难度 |
|------|------|----------|------|
| KV Cache INT8 量化 | 压缩 KV cache 体积，减少带宽需求 | GPT 可能从 7s → 5s | 需改 GPT 代码 |
| torch.compile() | 算子融合减少内存读写 | 不确定 | ROCm gfx1151 Triton 支持不完整 |
| MIOpen FIND_MODE 调优 | 强制搜索所有 solver 并缓存最优 | S2Mel 可能进一步提升 | 首次运行极慢（数小时） |
| 减少 CFM 步数 | 25 步 → 10-15 步 | S2Mel 2-2.5x 加速 | 可能影响音质 |

### 等待上游修复

- AMD 在 ROCm 中正式支持 gfx1151（添加到 MIOpen solver 的验证矩阵、生成 KDB）
- MIOpen 针对 RDNA3 优化更多卷积内核
- PyTorch ROCm 对 gfx11 系列的 torch.compile/Triton 支持完善

---

## 八、通用经验（适用于所有 gfx1151 AI 部署）

1. **始终设置 `HSA_OVERRIDE_GFX_VERSION=11.0.0`**：这是 gfx1151 运行任何 ROCm 应用的前提。

2. **始终设置 `amdgpu.cwsr_enable=0` 内核参数**：否则长时间 GPU 计算会间歇性崩溃。

3. **始终设置 `HSA_ENABLE_SDMA=0`**：避免数据传输 bug。

4. **卷积密集型模型考虑 CPU 运行**：如果模型以卷积为主（如 vocoder、U-Net decoder），在 gfx1151 上可能 CPU 更快（因为 MIOpen fallback）。判断标准：如果 GPU 运行时间异常长（比预期慢 10x+），试一下 CPU。

5. **FP16 优先尝试**：RDNA3 的 FP16 吞吐是 FP32 的 2 倍，对计算密集型操作效果显著，对带宽瓶颈操作也有帮助（数据体积减半）。

6. **torchvision 在 gfx1151 上不可用**：其 C++ 扩展会导致进程 abort。如果项目不直接使用 torchvision，直接卸载。如果间接依赖（如 transformers），需要 patch import 链。

7. **Docker build 阶段不能 import ROCm 的 C++ 扩展**：build 环境没有 GPU，`import torch` 可能触发 C++ 初始化失败。用 `importlib.metadata` 替代。

8. **pip 安装 torch 生态包必须从 ROCm 索引安装**：`--index-url https://download.pytorch.org/whl/rocm6.4`，否则会拉到 CUDA 版本。用 constraints 文件锁定。

9. **torchaudio 2.9+ 的 save 功能依赖 torchcodec**：在 ROCm 上不兼容，需用 soundfile 替代。

10. **混合设备部署需要注意 tensor device**：模型搬到不同设备后，调用时输入 tensor 必须也在同一设备上。内部有预计算 buffer（如 rotary embedding 的 freqs_cis、sequence mask）可能不跟着 `.to()` 走，导致 device mismatch。调用点越少、边界越清晰的模块越适合搬设备。
