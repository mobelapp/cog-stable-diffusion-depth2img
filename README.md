# cog-stable-diffusion-depth2img

[![Replicate](https://replicate.com/pwntus/stable-diffusion-depth2img/badge)](https://replicate.com/pwntus/stable-diffusion-depth2img)

This is an implementation of the [Diffusers Stable Diffusion v2](https://huggingface.co/stabilityai/stable-diffusion-2-depth) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="..." -i image=@...
