"""
Web UI for Low-VRAM Image Generation Tool
Uses Gradio to provide a simple interface for the generate.py script
Optimized to utilize high CPU memory (64GB) for offloading
"""

import os
import sys
import gradio as gr
import subprocess
import tempfile
from pathlib import Path
import time
import torch
import psutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from generate import check_system_resources
except ImportError:
    def check_system_resources():
        ram = psutil.virtual_memory()
        ram_total = ram.total / (1024 * 1024 * 1024)  # GB
        ram_available = ram.available / (1024 * 1024 * 1024)  # GB
        
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            allocated_vram = torch.cuda.memory_allocated(0) / (1024 * 1024)
            return allocated_vram, ram_available
        else:
            return 0, ram_available

def run_generation(
    prompt, 
    negative_prompt, 
    model_choice, 
    width, 
    height, 
    steps, 
    guidance, 
    seed, 
    lowvram, 
    precision, 
    cpu_offload,
    disable_xformers,
    scheduler,
    ref_images, 
    ref_strength,
    use_controlnet,
    controlnet_type,
    controlnet_image
):
    """Run the image generation script with the provided parameters including CPU optimization options"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "output.png")
        
        cmd = ["python", "generate.py"]
        
        if model_choice == "Stable Diffusion 1.5":
            cmd.append("--model")
            cmd.append("sd15")
        elif model_choice == "Stable Cascade":
            cmd.append("--model")
            cmd.append("stable-cascade")
        elif model_choice == "Stable Cascade Lite":
            cmd.append("--model")
            cmd.append("stable-cascade")
            cmd.append("--lite")
        
        cmd.extend(["--prompt", prompt])
        if negative_prompt:
            cmd.extend(["--negative_prompt", negative_prompt])
        
        cmd.extend(["--width", str(width)])
        cmd.extend(["--height", str(height)])
        cmd.extend(["--steps", str(steps)])
        cmd.extend(["--guidance", str(guidance)])
        
        if seed != -1:
            cmd.extend(["--seed", str(seed)])
        
        if lowvram:
            cmd.append("--lowvram")
        
        cmd.extend(["--precision", precision])
        cmd.extend(["--cpu_offload", cpu_offload])
        cmd.extend(["--scheduler", scheduler])
        
        if disable_xformers:
            cmd.append("--disable_xformers")
        
        if ref_images:
            for img in ref_images:
                if img is not None:
                    temp_img = os.path.join(temp_dir, f"ref_{Path(img).name}")
                    os.makedirs(os.path.dirname(temp_img), exist_ok=True)
                    
                    with open(img, "rb") as src, open(temp_img, "wb") as dst:
                        dst.write(src.read())
                    
                    cmd.extend(["--ref", temp_img])
            
            if ref_strength > 0:
                cmd.extend(["--ref_strength", str(ref_strength)])
        
        if use_controlnet and controlnet_image:
            cmd.append("--controlnet")
            cmd.extend(["--controlnet_type", controlnet_type])
            
            temp_img = os.path.join(temp_dir, f"control_{Path(controlnet_image).name}")
            os.makedirs(os.path.dirname(temp_img), exist_ok=True)
            
            with open(controlnet_image, "rb") as src, open(temp_img, "wb") as dst:
                dst.write(src.read())
            
            cmd.extend(["--controlnet_image", temp_img])
        
        cmd.extend(["--output", output_path])
        
        start_time = time.time()
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True
        )
        
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            output_lines.append(line.strip())
            yield output_path, "\n".join(output_lines)
        
        process.wait()
        
        elapsed_time = time.time() - start_time
        output_lines.append(f"Generation completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        
        if process.returncode != 0:
            output_lines.append(f"Error: Process exited with code {process.returncode}")
        
        if os.path.exists(output_path):
            yield output_path, "\n".join(output_lines)
        else:
            yield None, "\n".join(output_lines) + "\nError: Output image not generated"

def create_ui():
    """Create a simplified Gradio UI"""
    vram, available_ram = check_system_resources()
    
    gpu_info = "CPU only (CUDA not available)"
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)} ({total_vram:.0f}MB total, {vram:.0f}MB used)"
    
    ram_info = f"RAM: {available_ram:.1f}GB available"
    
    with gr.Blocks(title="Image Generator") as app:
        gr.Markdown("# 画像生成ツール / Image Generator")
        gr.Markdown(f"システム情報: {gpu_info} | {ram_info}")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="プロンプト / Prompt", 
                    lines=3, 
                    placeholder="画像の説明を入力してください / Describe the image you want to generate..."
                )
                
                negative_prompt = gr.Textbox(
                    label="ネガティブプロンプト / Negative Prompt", 
                    lines=2, 
                    placeholder="避けたい要素 / What you don't want in the image..."
                )
                
                with gr.Accordion("基本設定 / Basic Settings", open=True):
                    model_choice = gr.Dropdown(
                        label="モデル / Model", 
                        choices=["Stable Diffusion 1.5", "Stable Cascade Lite"],
                        value="Stable Diffusion 1.5"
                    )
                    
                    with gr.Row():
                        width = gr.Slider(label="幅 / Width", minimum=256, maximum=768, step=64, value=512)
                        height = gr.Slider(label="高さ / Height", minimum=256, maximum=768, step=64, value=512)
                
                with gr.Accordion("メモリ最適化 / Memory Optimization", open=True):
                    with gr.Row():
                        lowvram = gr.Checkbox(label="低VRAMモード / Low VRAM Mode", value=True)
                        precision = gr.Dropdown(
                            label="精度 / Precision", 
                            choices=["8bit", "4bit"], 
                            value="8bit"
                        )
                    
                    cpu_offload = gr.Radio(
                        label="CPUオフロード / CPU Offload", 
                        choices=["standard", "aggressive"], 
                        value="aggressive",
                        info="Aggressive: 64GB RAM推奨 / 64GB RAM recommended"
                    )
                
                with gr.Accordion("詳細設定 / Advanced Settings", open=False):
                    with gr.Row():
                        steps = gr.Slider(label="ステップ数 / Steps", minimum=10, maximum=50, step=1, value=30)
                        guidance = gr.Slider(label="ガイダンス / Guidance Scale", minimum=1.0, maximum=15.0, step=0.5, value=7.5)
                    
                    with gr.Row():
                        seed = gr.Number(label="シード値 / Seed (-1 for random)", value=-1, precision=0)
                        scheduler = gr.Dropdown(
                            label="スケジューラ / Scheduler", 
                            choices=["dpm++", "ddim", "euler_a"], 
                            value="dpm++"
                        )
                    
                    disable_xformers = gr.Checkbox(label="xformersを無効化 / Disable xformers", value=False)
                
                with gr.Accordion("参照画像 / Reference Images", open=False):
                    ref_images = gr.File(
                        label="参照画像（最大4枚） / Reference Images (up to 4)", 
                        file_count="multiple", 
                        file_types=["image"]
                    )
                    ref_strength = gr.Slider(
                        label="参照強度 / Reference Strength", 
                        minimum=0.0, 
                        maximum=1.0, 
                        step=0.05, 
                        value=0.5
                    )
                
                with gr.Accordion("ControlNet", open=False):
                    use_controlnet = gr.Checkbox(label="ControlNetを使用 / Use ControlNet", value=False)
                    controlnet_type = gr.Dropdown(
                        label="ControlNetタイプ / Type", 
                        choices=["canny", "depth", "pose"],
                        value="canny"
                    )
                    controlnet_image = gr.File(
                        label="ControlNet入力画像 / Input Image", 
                        file_types=["image"]
                    )
                
                generate_btn = gr.Button("画像を生成 / Generate Image", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_image = gr.Image(label="生成された画像 / Generated Image", type="filepath")
                output_text = gr.Textbox(label="生成ログ / Generation Log", lines=5)
        
        generate_btn.click(
            fn=run_generation,
            inputs=[
                prompt, negative_prompt, model_choice, width, height, 
                steps, guidance, seed, lowvram, precision,
                cpu_offload, disable_xformers, scheduler,
                ref_images, ref_strength, use_controlnet, controlnet_type, controlnet_image
            ],
            outputs=[output_image, output_text]
        )
    
    return app

if __name__ == "__main__":
    app = create_ui()
    app.queue()
    app.launch(share=True)  # Set share=True to create a public link
