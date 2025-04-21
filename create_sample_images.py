"""
Create sample reference images for the low-VRAM image generation tool
"""
from PIL import Image
import os

os.makedirs("examples/images", exist_ok=True)

blue_img = Image.new('RGB', (512, 512), color='blue')
blue_img.save('examples/images/reference2.jpg')

red_img = Image.new('RGB', (512, 512), color='red')
red_img.save('examples/images/reference3.jpg')

print("Sample reference images created successfully")
