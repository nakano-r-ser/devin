"""
Low-VRAM Image Generation Tool for GTX1650 (4GB VRAM)
Supports Stable Diffusion 1.5 with 8-bit/4-bit quantization and IP-Adapter for reference images
Optimized to utilize high CPU memory (64GB) for offloading
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import psutil
from pathlib import Path
from PIL import Image
from diffusers import (
    StableDiffusionPipeline, 
    DPMSolverMultistepScheduler,
    StableCascadeDecoderPipeline,
    StableCascadePriorPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler
)
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
import cv2

DEFAULT_MODEL = "stabilityai/stable-diffusion-1-5"
CASCADE_MODEL = "stabilityai/stable-cascade"
CASCADE_LITE_MODEL = "stabilityai/stable-cascade-lite"
IP_ADAPTER_MODEL = "InvokeAI/ip_adapter_plus_sd15"
CONTROLNET_CANNY = "lllyasviel/sd-controlnet-canny"
CONTROLNET_DEPTH = "lllyasviel/sd-controlnet-depth"
CONTROLNET_POSE = "lllyasviel/sd-controlnet-openpose"

def check_system_resources():
    """Check available VRAM and RAM"""
    ram = psutil.virtual_memory()
    ram_total = ram.total / (1024 * 1024 * 1024)  # GB
    ram_available = ram.available / (1024 * 1024 * 1024)  # GB
    ram_used = ram.used / (1024 * 1024 * 1024)  # GB
    ram_percent = ram.percent
    
    print(f"RAM: Total={ram_total:.1f}GB, Used={ram_used:.1f}GB, Available={ram_available:.1f}GB ({ram_percent}%)")
    
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        allocated_vram = torch.cuda.memory_allocated(0) / (1024 * 1024)
        cached_vram = torch.cuda.memory_reserved(0) / (1024 * 1024)
        
        print(f"VRAM: Total={total_vram:.0f}MB, Allocated={allocated_vram:.0f}MB, Cached={cached_vram:.0f}MB")
        return allocated_vram, ram_available
    else:
        print("CUDA not available, running on CPU only")
        return 0, ram_available

def optimize_for_cpu_offload(high_ram=False):
    """Configure PyTorch for optimal CPU offloading based on available RAM"""
    if high_ram:
        torch.set_num_threads(min(16, os.cpu_count()))
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    else:
        torch.set_num_threads(min(8, os.cpu_count()))
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    return {
        "num_threads": torch.get_num_threads(),
        "max_split_size": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")
    }

def load_ip_adapter(pipe, ip_adapter_path, device, dtype):
    """Load IP-Adapter-Plus for reference image conditioning"""
    from ip_adapter.ip_adapter_plus import IPAdapterPlus
    
    if not os.path.exists(ip_adapter_path):
        os.makedirs(os.path.dirname(ip_adapter_path), exist_ok=True)
        hf_hub_download(
            repo_id=IP_ADAPTER_MODEL,
            filename="ip_adapter_plus_sd15.bin",
            local_dir=os.path.dirname(ip_adapter_path)
        )
    
    ip_adapter = IPAdapterPlus(pipe, ip_adapter_path, device=device, dtype=dtype)
    return ip_adapter

def process_reference_images(ip_adapter, ref_images, ref_strength):
    """Process reference images with IP-Adapter"""
    images = [load_image(img_path) for img_path in ref_images]
    return ip_adapter.set_images(images, strength=ref_strength)

def prepare_controlnet(controlnet_type, image_path, low_vram=False):
    """Prepare ControlNet input image and model"""
    image = load_image(image_path)
    
    if controlnet_type == "canny":
        model_id = CONTROLNET_CANNY
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
    
    elif controlnet_type == "depth":
        model_id = CONTROLNET_DEPTH
        if image_path.endswith((".png", ".jpg", ".jpeg")):
            pass  # Use as is
    
    elif controlnet_type == "pose":
        model_id = CONTROLNET_POSE
        if image_path.endswith((".png", ".jpg", ".jpeg")):
            pass  # Use as is
    
    else:
        raise ValueError(f"Unsupported ControlNet type: {controlnet_type}")
    
    controlnet = ControlNetModel.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    return controlnet, image

def load_stable_diffusion(args):
    """Load Stable Diffusion model with memory optimizations"""
    initial_vram, available_ram = check_system_resources()
    
    high_ram = available_ram > 32  # Consider high RAM if more than 32GB available
    cpu_config = optimize_for_cpu_offload(high_ram)
    print(f"Configured for {'high' if high_ram else 'standard'} RAM usage: {cpu_config}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    if args.precision == "4bit":
        from bitsandbytes.nn import Linear4bit
        quantization_config = {"load_in_4bit": True}
    else:  # 8bit
        quantization_config = {"load_in_8bit": True}
    
    if args.controlnet:
        controlnet, control_image = prepare_controlnet(
            args.controlnet_type, args.controlnet_image, args.lowvram
        )
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            DEFAULT_MODEL,
            controlnet=controlnet,
            torch_dtype=dtype,
            **quantization_config
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            DEFAULT_MODEL,
            torch_dtype=dtype,
            **quantization_config
        )
    
    if args.lowvram or device == "cuda":
        if high_ram and getattr(args, 'cpu_offload', '') == "aggressive":
            from accelerate import cpu_offload_with_hook
            
            for component in [pipe.unet, pipe.vae, pipe.text_encoder]:
                cpu_offload_with_hook(component, device, offload_buffers=True)
            
            print("Using aggressive CPU offloading (high RAM mode)")
        else:
            # Standard CPU offloading
            pipe.enable_model_cpu_offload()
            print("Using standard CPU offloading")
        
        pipe.enable_vae_slicing()
        pipe.enable_attention_slicing(slice_size="auto")
        
        if hasattr(pipe, "enable_xformers_memory_efficient_attention") and not getattr(args, 'disable_xformers', False):
            pipe.enable_xformers_memory_efficient_attention()
            print("Using xformers for memory-efficient attention")
    
    scheduler_type = getattr(args, 'scheduler', 'dpm++')
    if scheduler_type == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "euler_a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    else:  # dpm++
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
    
    loaded_vram, _ = check_system_resources()
    vram_used = loaded_vram - initial_vram
    print(f"Model loaded, using {vram_used:.0f}MB of VRAM")
    
    if loaded_vram > 3800 and device == "cuda":
        print(f"ERROR: VRAM usage too high ({loaded_vram:.0f}MB > 3800MB)")
        print("Try using --lowvram or --precision 4bit or --cpu_offload aggressive")
        sys.exit(1)
    
    return pipe, device, dtype

def load_stable_cascade(args):
    """Load Stable Cascade model with memory optimizations"""
    initial_vram, available_ram = check_system_resources()
    
    high_ram = available_ram > 32  # Consider high RAM if more than 32GB available
    cpu_config = optimize_for_cpu_offload(high_ram)
    print(f"Configured for {'high' if high_ram else 'standard'} RAM usage: {cpu_config}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    model_id = CASCADE_LITE_MODEL if args.lite else CASCADE_MODEL
    
    prior = StableCascadePriorPipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        variant="bf16" if torch.cuda.is_available() else None
    )
    
    decoder = StableCascadeDecoderPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="bf16" if torch.cuda.is_available() else None
    )
    
    if high_ram and getattr(args, 'cpu_offload', '') == "aggressive":
        from accelerate import cpu_offload_with_hook
        
        for component in [prior.prior, prior.text_encoder]:
            cpu_offload_with_hook(component, device, offload_buffers=True)
        
        for component in [decoder.decoder, decoder.text_encoder]:
            cpu_offload_with_hook(component, device, offload_buffers=True)
        
        print("Using aggressive CPU offloading for Stable Cascade (high RAM mode)")
    else:
        # Standard CPU offloading
        prior.enable_model_cpu_offload()
        decoder.enable_model_cpu_offload()
        print("Using standard CPU offloading for Stable Cascade")
    
    loaded_vram, _ = check_system_resources()
    vram_used = loaded_vram - initial_vram
    print(f"Stable Cascade loaded, using {vram_used:.0f}MB of VRAM")
    
    if loaded_vram > 3800 and device == "cuda":
        print(f"ERROR: VRAM usage too high ({loaded_vram:.0f}MB > 3800MB)")
        print("Try using --lite for Stable Cascade Lite or --cpu_offload aggressive")
        sys.exit(1)
    
    return prior, decoder, device, dtype

def generate_image(args):
    """Generate image based on command line arguments"""
    start_time = time.time()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if args.model == "stable-cascade":
        prior, decoder, device, dtype = load_stable_cascade(args)
        
        print(f"Generating with Stable Cascade using prompt: {args.prompt}")
        prior_output = prior(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance,
            num_inference_steps=args.prior_steps or 30,
            num_images_per_prompt=1,
        )
        
        decoder_output = decoder(
            image_embeddings=prior_output.image_embeddings,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            output_type="pil",
        )
        image = decoder_output.images[0]
        
    else:  # Stable Diffusion
        pipe, device, dtype = load_stable_diffusion(args)
        
        if args.ref:
            ip_adapter_path = os.path.expanduser("~/.cache/ip_adapter/ip_adapter_plus_sd15.bin")
            ip_adapter = load_ip_adapter(pipe, ip_adapter_path, device, dtype)
            process_reference_images(ip_adapter, args.ref, args.ref_strength)
            
            print(f"Generating with IP-Adapter using prompt: {args.prompt}")
            print(f"Reference images: {args.ref}")
            
            image = ip_adapter.generate(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_samples=1,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed,
            )[0]
            
        elif args.controlnet:
            controlnet, control_image = prepare_controlnet(
                args.controlnet_type, args.controlnet_image, args.lowvram
            )
            
            print(f"Generating with ControlNet ({args.controlnet_type}) using prompt: {args.prompt}")
            image = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=control_image,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                width=args.width,
                height=args.height,
                generator=torch.Generator(device).manual_seed(args.seed) if args.seed != -1 else None,
            ).images[0]
            
        else:
            # Standard generation
            print(f"Generating with Stable Diffusion using prompt: {args.prompt}")
            image = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                width=args.width,
                height=args.height,
                generator=torch.Generator(device).manual_seed(args.seed) if args.seed != -1 else None,
            ).images[0]
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    image.save(args.output)
    print(f"Image saved to {args.output}")
    
    elapsed_time = time.time() - start_time
    print(f"Generation completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    
    check_system_resources()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return image

def main():
    parser = argparse.ArgumentParser(description="Low-VRAM Image Generation Tool")
    
    parser.add_argument("--model", type=str, default="sd15", choices=["sd15", "stable-cascade"],
                        help="Model to use: sd15 (Stable Diffusion 1.5) or stable-cascade")
    parser.add_argument("--lite", action="store_true", help="Use Stable Cascade Lite (smaller model)")
    
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", 
                        help="Negative prompt to specify what you don't want")
    
    parser.add_argument("--width", type=int, default=512, help="Image width (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Image height (default: 512)")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps (default: 30)")
    parser.add_argument("--prior_steps", type=int, default=None, 
                        help="Number of prior inference steps for Stable Cascade (default: 30)")
    parser.add_argument("--guidance", type=float, default=7.5, 
                        help="Guidance scale (default: 7.5)")
    parser.add_argument("--seed", type=int, default=-1, 
                        help="Random seed (-1 for random)")
    parser.add_argument("--scheduler", type=str, default="dpm++", choices=["dpm++", "ddim", "euler_a"],
                        help="Scheduler to use (default: dpm++)")
    
    parser.add_argument("--lowvram", action="store_true", 
                        help="Enable aggressive memory optimization for low VRAM GPUs")
    parser.add_argument("--precision", type=str, default="8bit", choices=["8bit", "4bit"],
                        help="Model precision: 8bit or 4bit (default: 8bit)")
    parser.add_argument("--cpu_offload", type=str, default="standard", choices=["standard", "aggressive"],
                        help="CPU offloading strategy (aggressive uses more RAM but less VRAM)")
    parser.add_argument("--disable_xformers", action="store_true",
                        help="Disable xformers memory efficient attention")
    
    # Reference images
    parser.add_argument("--ref", action="append", default=None,
                        help="Reference image path (can be used multiple times)")
    parser.add_argument("--ref_strength", type=float, default=0.5,
                        help="Strength of reference image influence (0-1, default: 0.5)")
    
    # ControlNet
    parser.add_argument("--controlnet", action="store_true", help="Use ControlNet")
    parser.add_argument("--controlnet_type", type=str, default="canny",
                        choices=["canny", "depth", "pose"],
                        help="ControlNet type: canny, depth, or pose (default: canny)")
    parser.add_argument("--controlnet_image", type=str, default=None,
                        help="Input image for ControlNet")
    
    args = parser.parse_args()
    
    initial_vram, available_ram = check_system_resources()
    
    if args.width * args.height > 512 * 768 and not args.lowvram:
        print("Warning: Large image size may require more VRAM. Consider using --lowvram.")
    
    if args.controlnet and not args.controlnet_image:
        parser.error("--controlnet requires --controlnet_image")
    
    # Optimize for high RAM if available
    if available_ram > 32 and args.cpu_offload == "aggressive":
        print(f"High RAM detected ({available_ram:.1f}GB). Using aggressive CPU offloading.")
    elif available_ram < 16 and args.cpu_offload == "aggressive":
        print("Warning: Aggressive CPU offloading requested but RAM is limited. May cause system slowdown.")
    
    generate_image(args)

if __name__ == "__main__":
    main()
