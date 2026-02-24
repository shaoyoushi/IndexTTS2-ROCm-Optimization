# IndexTTS2 with ROCm for AMD Strix Halo (gfx1151)
# Base: ROCm 6.4 + PyTorch (official AMD image)
FROM rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ---- ROCm environment for gfx1151 (Strix Halo) ----
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV GPU_MAX_ALLOC_PERCENT=100
ENV GPU_MAX_HEAP_SIZE=100
ENV HSA_ENABLE_SDMA=0
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
ENV MIOPEN_LOG_LEVEL=4

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

# ---- Clone IndexTTS2 (skip LFS - repo LFS budget exceeded) ----
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/index-tts/index-tts.git . \
    && echo ">> Git clone done (LFS files skipped, not needed for inference)"

# ---- Install dependencies ----
# Strategy:
#   1. Relax pyproject.toml's torch version (==2.8.* -> >=2.7.0)
#   2. Create pip constraints file to LOCK our ROCm torch/torchaudio
#   3. pip install ".[webui]" reads pyproject.toml for correct deps
#   4. Constraints prevent pip from replacing ROCm packages with CUDA versions
#
# IMPORTANT: Do NOT install torchvision. Its C++ extension (_C.so) crashes on
# gfx1151 with "basic_string::_M_construct null not valid". IndexTTS2 doesn't
# need it; only transformers imports it indirectly for object detection losses.

# Step 1: Patch pyproject.toml for ROCm + Python 3.12 compatibility
RUN sed -i 's/"torch==2.8.\*"/"torch>=2.7.0"/' pyproject.toml \
    && sed -i 's/"torchaudio==2.8.\*"/"torchaudio>=2.7.0"/' pyproject.toml \
    && sed -i 's/"numba==0.58.1"/"numba>=0.60.0"/' pyproject.toml

# Step 2: Install ROCm-compatible torchaudio (base image only has torch)
#   Without ROCm version, pip pulls CUDA torchaudio from PyPI which links
#   to libtorch_cuda.so and crashes on import.
RUN pip install --no-cache-dir torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.4

# Step 3: Generate constraints to lock ROCm torch + torchaudio
#   Use importlib.metadata (not import torch) to avoid C++ init during build.
RUN echo 'from importlib.metadata import version' > /tmp/gen_constraints.py \
    && echo 'for pkg in ["torch", "torchaudio"]:' >> /tmp/gen_constraints.py \
    && echo '    v = version(pkg).split("+")[0]' >> /tmp/gen_constraints.py \
    && echo '    print(f"{pkg}=={v}")' >> /tmp/gen_constraints.py \
    && python /tmp/gen_constraints.py > /tmp/torch-constraints.txt \
    && echo ">> Torch constraints:" && cat /tmp/torch-constraints.txt

# Step 4: Install project + all deps, constrained to keep ROCm packages
RUN pip install --no-cache-dir -c /tmp/torch-constraints.txt ".[webui]"

# Step 5: Uninstall torchvision if pip pulled it as a transitive dependency
#   Its _C.so triggers a C++ abort on gfx1151 when imported. IndexTTS2 does
#   not use it directly. transformers has is_vision_available() guards.
RUN pip uninstall torchvision -y 2>/dev/null; true

# Step 6: Patch transformers to avoid importing torchvision-dependent modules
#   transformers.loss.loss_utils imports object detection losses that pull in
#   torchvision. Comment out these imports and their LOSS_MAPPING entries.
RUN LOSS_UTILS=$(python -c "import transformers.loss; import os; print(os.path.join(os.path.dirname(transformers.loss.__file__), 'loss_utils.py'))") \
    && sed -i 's/^from .loss_d_fine/#from .loss_d_fine/' "$LOSS_UTILS" \
    && sed -i 's/^from .loss_deformable_detr/#from .loss_deformable_detr/' "$LOSS_UTILS" \
    && sed -i 's/^from .loss_for_object_detection/#from .loss_for_object_detection/' "$LOSS_UTILS" \
    && sed -i 's/^from .loss_grounding_dino/#from .loss_grounding_dino/' "$LOSS_UTILS" \
    && sed -i 's/^from .loss_rt_detr/#from .loss_rt_detr/' "$LOSS_UTILS" \
    && sed -i '/"ForSegmentation":/d' "$LOSS_UTILS" \
    && sed -i '/"ForObjectDetection":/d' "$LOSS_UTILS" \
    && sed -i '/"DeformableDetrForObjectDetection":/d' "$LOSS_UTILS" \
    && sed -i '/"ConditionalDetrForObjectDetection":/d' "$LOSS_UTILS" \
    && sed -i '/"DabDetrForObjectDetection":/d' "$LOSS_UTILS" \
    && sed -i '/"GroundingDinoForObjectDetection":/d' "$LOSS_UTILS" \
    && sed -i '/"ConditionalDetrForSegmentation":/d' "$LOSS_UTILS" \
    && sed -i '/"RTDetrForObjectDetection":/d' "$LOSS_UTILS" \
    && sed -i '/"RTDetrV2ForObjectDetection":/d' "$LOSS_UTILS" \
    && sed -i '/"DFineForObjectDetection":/d' "$LOSS_UTILS" \
    && echo ">> Patched transformers loss_utils.py"

# Step 7: Patch torchaudio.save -> soundfile
#   torchaudio 2.9.x defaults to torchcodec backend which is incompatible
#   with ROCm PyTorch. Use soundfile (already installed) for WAV output.
RUN sed -i 's/torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)/import soundfile as sf; sf.write(output_path, wav.type(torch.int16).squeeze().cpu().numpy(), sampling_rate)/' indextts/infer_v2.py \
    && echo ">> Patched infer_v2.py torchaudio.save -> soundfile"

# Step 8: Hybrid device mode — BigVGAN on CPU, everything else on GPU
#   gfx1151 MIOpen has no optimized kernels for BigVGAN's conv layers,
#   causing 82s for a 10s clip. CPU (Zen5 16-core) does it in ~1s.
#   GPT and S2Mel stay on GPU where they are 3-6x faster than CPU.
#
#   Patch 1: infer_v2.py — add .cpu() before BigVGAN forward call
RUN sed -i 's/wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)/wav = self.bigvgan(vc_target.float().cpu()).squeeze().unsqueeze(0)/' indextts/infer_v2.py \
    && echo ">> Patched infer_v2.py BigVGAN input to CPU"
#   Patch 2: webui.py — move BigVGAN to CPU (float32) after model init
#   .float() ensures BigVGAN stays fp32 on CPU even when --fp16 is enabled
RUN sed -i '/^# 支持的语言列表/i\tts.bigvgan = tts.bigvgan.to("cpu").float()  # Hybrid mode: BigVGAN on CPU fp32 for 9x speedup on gfx1151\nprint(">> BigVGAN moved to CPU float32 (hybrid device mode)")' webui.py \
    && echo ">> Patched webui.py BigVGAN -> CPU float32"

# Step 9: Verify packages survived correctly
RUN python -c "from importlib.metadata import version as v; [print(f'{p}=={v(p)}') for p in ['torch','torchaudio','gradio','transformers','librosa','numpy']]; print('All packages installed OK')" \
    && python -c "v=__import__('importlib.metadata',fromlist=['version']).version('torch'); assert 'rocm' in v or 'git' in v, f'FATAL: torch {v} not ROCm!'; print(f'ROCm torch verified: {v}')"

# ---- Download model checkpoints ----
RUN pip install --no-cache-dir "huggingface_hub[hf_xet]" \
    && python -c "from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2', local_dir='checkpoints')"

# ---- Expose WebUI port ----
EXPOSE 8765

# ---- Launch ----
# No --cuda_kernel: BigVGAN CUDA kernel is NVIDIA-compiled, incompatible with ROCm
# No --deepspeed: untested on ROCm
# --fp16: 2.4x speedup on S2Mel, GPT uses half precision, BigVGAN stays fp32 on CPU
CMD ["python", "webui.py", "--host", "0.0.0.0", "--port", "8765", "--fp16"]
