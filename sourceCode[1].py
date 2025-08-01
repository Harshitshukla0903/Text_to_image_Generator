import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
from contextlib import nullcontext

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch_dtype
).to(device)

pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

def generate_image(prompt):
    try:
        with torch.autocast(device_type=device) if device == "cuda" else nullcontext():
            result = pipe(prompt)
            return result.images[0]
    except Exception as e:
        print(f"Error: {e}")
        return f"Error generating image: {e}"

iface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, label="Prompt", placeholder="e.g. A futuristic sports car in a neon-lit city..."),
    outputs="image",
    title="Text-to-Image Generator (Stable Diffusion)",
    description="⚡ Generate images locally using Stable Diffusion v1.5. Make sure your GPU (if available) is utilized properly."
)

iface.launch()


