from typing import Optional
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, UniPCMultistepScheduler


class SDXLDepthControlNet:
    def __init__(
        self,
        controlnet_id: str = "diffusers/controlnet-depth-sdxl-1.0",
        sdxl_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = None,
        use_cpu_offload: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            sdxl_id,
            controlnet=self.controlnet,
            torch_dtype=dtype,
            safety_checker=None
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        if self.device == "cuda" and not use_cpu_offload:
            self.pipe.to("cuda")
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
        else:
            self.pipe.to("cuda" if self.device == "cuda" else "cpu")
            if self.device == "cuda" and use_cpu_offload:
                self.pipe.enable_model_cpu_offload()

    @torch.no_grad()
    def generate(
        self,
        depth_control_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        out_size: int = 1024,
        seed: int = 0,
        steps: int = 35,
        guidance: float = 7.0,
        control_scale: float = 1.0,
    ) -> Image.Image:
        depth_control_image = depth_control_image.resize((out_size, out_size)).convert("RGB")

        gen = torch.Generator(device="cuda" if self.device == "cuda" else "cpu").manual_seed(seed)

        img = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=depth_control_image,
            controlnet_conditioning_scale=control_scale,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gen,
            height=out_size,
            width=out_size
        ).images[0]

        return img
