# Low-VRAM Image Generation Tool

A lightweight image generation tool designed specifically for GPUs with limited VRAM (4GB or less), such as the GTX1650. This tool uses quantized models and memory optimization techniques to enable high-quality image generation on consumer hardware.

## Features

- **Low VRAM Usage**: Generate images with as little as 4GB VRAM
- **Multiple Models**: 
  - Stable Diffusion 1.5 (default, quantized to 8-bit or 4-bit)
  - Stable Cascade Lite (optional, more recent model)
- **Reference Image Support**: Blend styles from multiple reference images using IP-Adapter-Plus
- **ControlNet Support**: Optional control over image generation with Canny edge, depth, or pose conditioning
- **Memory Optimizations**: CPU offloading, attention slicing, VAE slicing, and xformers
- **Full Prompt Control**: Detailed positive and negative prompts

## Quick Start

### Installation

#### Windows

```bash
# Install Miniconda if you don't have it
# Create and activate environment
conda env create -f environment.yml
conda activate sd-lowvram

# Generate your first image
python generate.py --prompt "a beautiful landscape with mountains and a lake" --output landscape.png
```

#### Linux

```bash
# Install dependencies
conda env create -f environment.yml
conda activate sd-lowvram

# Generate your first image
python generate.py --prompt "a beautiful landscape with mountains and a lake" --output landscape.png
```

#### macOS

```bash
# Install dependencies (Note: CUDA not available on macOS, will run on CPU)
conda env create -f environment.yml
conda activate sd-lowvram

# Generate your first image (will be slow on CPU)
python generate.py --prompt "a beautiful landscape with mountains and a lake" --output landscape.png
```

## Usage Examples

### Basic Image Generation

```bash
python generate.py --prompt "a beautiful landscape with mountains and a lake" --output landscape.png
```

### Using Negative Prompts

```bash
python generate.py --prompt "a beautiful landscape with mountains and a lake" --negative_prompt "ugly, blurry, low quality" --output landscape.png
```

### Using Reference Images

```bash
python generate.py --prompt "portrait of a person" --ref reference1.jpg --ref reference2.jpg --ref reference3.jpg --ref_strength 0.7 --output portrait.png
```

### Using ControlNet

```bash
python generate.py --prompt "portrait of a person" --controlnet --controlnet_type canny --controlnet_image input.jpg --output controlled.png
```

### Low VRAM Mode

```bash
python generate.py --prompt "a beautiful landscape" --lowvram --precision 4bit --output landscape_lowvram.png
```

### Using Stable Cascade

```bash
python generate.py --prompt "a beautiful landscape" --model stable-cascade --lite --output landscape_cascade.png
```

## VRAM Optimization Tips

1. **Resolution**: Start with 512×512 resolution for best performance
2. **Precision**: Use `--precision 4bit` for maximum VRAM savings
3. **Low VRAM Mode**: Always use `--lowvram` on 4GB cards
4. **Steps**: Reduce steps (e.g., `--steps 20`) for faster generation
5. **Clear Cache**: The tool automatically clears CUDA cache, but you may need to restart for multiple generations
6. **Monitor Usage**: Check VRAM usage with `nvidia-smi` while generating

## CPU and GPU Collaboration (CPU/GPUの連携について)

This tool optimizes image generation by intelligently distributing workloads between GPU and CPU memory:

```
CPUとGPUを連携させた画像生成の仕組み:

1. GPUメモリ(VRAM)の最適化:
   - モデルの量子化 (8ビットまたは4ビット精度に圧縮)
   - アテンションメカニズムの最適化 (xformers, スライシング)
   - VAEのCPUオフロード (画像エンコード/デコード処理)

2. 大容量CPUメモリ(64GB)の活用:
   - 「aggressive」CPUオフロードモード: モデルの一部をCPUメモリに配置
   - 動的メモリ管理: 利用可能なRAMに基づいてスレッド数と割り当てを最適化
   - PyTorchのキャッシュ管理: 不要なメモリを定期的に解放

3. 処理フロー:
   - プロンプト解析とトークン化: CPU上で実行
   - 参照画像処理: CPU上でIP-Adapterを使用して処理
   - 主要な推論計算: GPU上で実行 (必要に応じてCPUにオフロード)
   - 最終画像生成: CPUとGPUの両方を使用

4. 最適化設定:
   - 標準モード: 基本的なCPUオフロード (16GB RAM以上推奨)
   - アグレッシブモード: 大量のCPUオフロード (64GB RAM推奨、VRAM使用量を最小化)
```

The `--cpu_offload aggressive` option maximizes the use of available CPU memory (up to 64GB) to minimize VRAM usage, making it possible to run larger models on limited GPUs like the GTX1650.

## Command Line Options

```
usage: generate.py [-h] [--model {sd15,stable-cascade}] [--lite] --prompt PROMPT [--negative_prompt NEGATIVE_PROMPT] [--width WIDTH]
                  [--height HEIGHT] [--output OUTPUT] [--steps STEPS] [--prior_steps PRIOR_STEPS] [--guidance GUIDANCE] [--seed SEED]
                  [--lowvram] [--precision {8bit,4bit}] [--ref REF] [--ref_strength REF_STRENGTH] [--controlnet]
                  [--controlnet_type {canny,depth,pose}] [--controlnet_image CONTROLNET_IMAGE]

Low-VRAM Image Generation Tool

options:
  -h, --help            show this help message and exit
  --model {sd15,stable-cascade}
                        Model to use: sd15 (Stable Diffusion 1.5) or stable-cascade
  --lite                Use Stable Cascade Lite (smaller model)
  --prompt PROMPT       Text prompt for image generation
  --negative_prompt NEGATIVE_PROMPT
                        Negative prompt to specify what you don't want
  --width WIDTH         Image width (default: 512)
  --height HEIGHT       Image height (default: 512)
  --output OUTPUT       Output image path
  --steps STEPS         Number of inference steps (default: 30)
  --prior_steps PRIOR_STEPS
                        Number of prior inference steps for Stable Cascade (default: 30)
  --guidance GUIDANCE   Guidance scale (default: 7.5)
  --seed SEED           Random seed (-1 for random)
  --lowvram             Enable aggressive memory optimization for low VRAM GPUs
  --precision {8bit,4bit}
                        Model precision: 8bit or 4bit (default: 8bit)
  --ref REF             Reference image path (can be used multiple times)
  --ref_strength REF_STRENGTH
                        Strength of reference image influence (0-1, default: 0.5)
  --controlnet          Use ControlNet
  --controlnet_type {canny,depth,pose}
                        ControlNet type: canny, depth, or pose (default: canny)
  --controlnet_image CONTROLNET_IMAGE
                        Input image for ControlNet
```

## Requirements

- Python 3.11
- PyTorch 2.3.0 with CUDA 11.x
- NVIDIA GPU with at least 4GB VRAM (GTX1650 or better)
- For full requirements, see `environment.yml`
