import os
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionDepth2ImgPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from PIL import Image

MODEL_ID = "stabilityai/stable-diffusion-2-depth"
MODEL_CACHE = "diffusers-cache"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            revision='fp16',
            torch_dtype=torch.float16,
        ).to("cuda")

        # Needs xformers
        # self.pipe.enable_xformers_memory_efficient_attention()
        # self.pipe.enable_vae_slicing()

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="The prompt to guide the image generation.",
            default="A fantasy landscape, trending on artstation",
        ),
        negative_prompt: str = Input(
            description="The prompt NOT to guide the image generation. Ignored when not using guidance",
            default=None,
        ),
        image: Path = Input(
            description="Image that will be used as the starting point for the process.",
        ),
        prompt_strength: float = Input(
            description="Prompt strength when providing the image. 1.0 corresponds to full destruction of information in init image.",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output. Higher number of outputs may OOM.",
            ge=1,
            le=8,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
            ge=1,
            le=500,
            default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance. Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.",
            ge=1,
            le=20,
            default=7.5
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)

        extra_kwargs = {
            "image": Image.open(image).convert("RGB"),
            "strength": prompt_strength,
        }

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_outputs,
            num_inference_steps=num_inference_steps,
            **extra_kwargs,

            # This will OOM if num_outputs > 1
            # prompt=[prompt] * num_outputs if prompt is not None else None,
            # negative_prompt=[negative_prompt] * num_outputs if negative_prompt is not None else None,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]

