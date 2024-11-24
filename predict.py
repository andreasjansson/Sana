# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
import torch
from torchvision.utils import save_image
from cog import BasePredictor, Input, Path

from app.sana_pipeline import SanaPipeline


MODEL_CACHE = "model_cache"
MODEL_URL = f"https://weights.replicate.delivery/default/NVlabs/Sana/{MODEL_CACHE}.tar"

os.environ.update(
    {
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": MODEL_CACHE,
        "TORCH_HOME": MODEL_CACHE,
        "HF_DATASETS_CACHE": MODEL_CACHE,
        "TRANSFORMERS_CACHE": MODEL_CACHE,
        "HUGGINGFACE_HUB_CACHE": MODEL_CACHE,
    }
)


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        config_path = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml"
        self.pipe = SanaPipeline(config_path)
        model_path = f"{MODEL_CACHE}/Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth"
        self.pipe.from_pretrained(model_path)

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default='a cyberpunk cat with a neon sign that says "Sana"',
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="",
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_inference_steps: int = Input(description="Number of denoising steps", ge=1, default=18),
        guidance_scale: float = Input(description="Classifier-free guidance scale", ge=1, le=20, default=5.0),
        pag_guidance_scale: float = Input(description="PAG Guidance scale", ge=1, le=20, default=2.0),
        seed: int = Input(description="Random seed. Leave blank to randomize the seed", default=None),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            pag_guidance_scale=pag_guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        out_path = "/tmp/output.png"
        save_image(image, out_path, nrow=1, normalize=True, value_range=(-1, 1))
        return Path(out_path)
