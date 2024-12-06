from dataclasses import dataclass
import os
import subprocess
import time
import torch
from torchvision.utils import save_image
from cog import BasePredictor, Input, Path

from app.sana_pipeline import SanaPipeline


@dataclass(frozen=True)
class ModelVariant:
    dir: str
    checkpoint: str
    config: str


MODEL_URL = "https://weights.replicate.delivery/default/nvidia/sana/sana.tar"
MODEL_ROOT_DIR = Path("/weights")
VARIANTS = {
    "1600M-1024px": ModelVariant(
        "Sana_1600M_1024px", "Sana_1600M_1024px.pth", "1024ms/Sana_1600M_img1024.yaml"
    ),
    "1600M-1024px-multilang": ModelVariant(
        "Sana_1600M_1024px_MultiLing", "Sana_1600M_1024px_MultiLing.pth", "1024ms/Sana_1600M_img1024.yaml"
    ),
    "1600M-512px": ModelVariant(
        "Sana_1600M_512px", "Sana_1600M_512px.pth", "512ms/Sana_1600M_img512.yaml"
    ),
    "600M-1024px-multilang": ModelVariant(
        "Sana_600M_1024px", "Sana_600M_1024px_MultiLing.pth", "1024ms/Sana_600M_img1024.yaml"
    ),
    "600M-512px-multilang": ModelVariant(
        "Sana_600M_512px", "Sana_600M_512px_MultiLing.pth", "512ms/Sana_600M_img512.yaml"
    ),
}


def download_weights(url: str, dest: Path):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, str(dest)], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.pipes = {}

        if not MODEL_ROOT_DIR.exists():
            download_weights(MODEL_URL, MODEL_ROOT_DIR)

        vae = tokenizer = text_encoder = None
        for name, variant in VARIANTS.items():
            model_dir = MODEL_ROOT_DIR / "sana" / variant.dir
            config_path = f"configs/sana_config/{variant.config}"
            pipe = SanaPipeline(config_path, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder)
            vae = pipe.vae
            tokenizer = pipe.tokenizer
            text_encoder = pipe.text_encoder
            checkpoint_path = model_dir / "checkpoints" / variant.checkpoint
            print(f"Loading {name} from {checkpoint_path}")
            pipe.from_pretrained(str(checkpoint_path))
            self.pipes[name] = pipe

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
        model_variant: str = Input(
            description="Model variant. 1600M variants are slower but produce higher quality than 600M, 1024px variants are optimized for 1024x1024px images, 512px variants are optimized for 512x512px images, 'multilang' variants can be prompted in both English and Chinese",
            choices=sorted(VARIANTS.keys()),
            default="1600M-1024px"
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, default=18
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale", ge=1, le=20, default=5.0
        ),
        pag_guidance_scale: float = Input(
            description="PAG Guidance scale", ge=1, le=20, default=2.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(seed)

        pipe = self.pipes[model_variant]
        image = pipe(
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
