"""BiRefNet background removal for InvokeAI."""

import importlib.util
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Literal

import torch
from PIL import Image
from torchvision import transforms

from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    InputField,
    InvocationContext,
    invocation,
)

from invokeai.app.invocations.image import ImageOutput
from invokeai.backend.util.devices import TorchDevice


MODELS = {
    "BiRefNet": "ZhengPeng7/BiRefNet",
    "BiRefNet-HR": "ZhengPeng7/BiRefNet_HR",
    "BiRefNet-matting": "ZhengPeng7/BiRefNet-matting",
    "BiRefNet-HR-matting": "ZhengPeng7/BiRefNet_HR-matting",
    "BiRefNet-portrait": "ZhengPeng7/BiRefNet-portrait",
    "BiRefNet-512": "ZhengPeng7/BiRefNet_512x512",
    "BiRefNet-dynamic": "ZhengPeng7/BiRefNet_dynamic",
    "BiRefNet-dynamic-matting": "ZhengPeng7/BiRefNet_dynamic-matting",
    "BiRefNet-lite": "ZhengPeng7/BiRefNet_lite",
    "BiRefNet-lite-2K": "ZhengPeng7/BiRefNet_lite-2K",
    "BiRefNet-lite-matting": "ZhengPeng7/BiRefNet_lite-matting",
    "BiRefNet-HRSOD": "ZhengPeng7/BiRefNet-HRSOD",
    "BiRefNet-DIS5K": "ZhengPeng7/BiRefNet-DIS5K",
    "BiRefNet-COD": "ZhengPeng7/BiRefNet-COD",
}

ModelType = Literal[
    "BiRefNet",
    "BiRefNet-HR",
    "BiRefNet-matting",
    "BiRefNet-HR-matting",
    "BiRefNet-portrait",
    "BiRefNet-512",
    "BiRefNet-dynamic",
    "BiRefNet-dynamic-matting",
    "BiRefNet-lite",
    "BiRefNet-lite-2K",
    "BiRefNet-lite-matting",
    "BiRefNet-HRSOD",
    "BiRefNet-DIS5K",
    "BiRefNet-COD",
]

_MODEL_CACHE = {}
_IMAGE_PROC_MODULE = None


def _get_model(model_name: str, device: torch.device) -> torch.nn.Module:
    cache_key = (model_name, str(device))

    if cache_key not in _MODEL_CACHE:
        from transformers import AutoModelForImageSegmentation

        repo_id = MODELS[model_name]
        model = AutoModelForImageSegmentation.from_pretrained(
            repo_id, trust_remote_code=True
        )
        model.eval()
        model.to(device)
        if device.type == "cuda":
            model.half()

        _MODEL_CACHE[cache_key] = model

    return _MODEL_CACHE[cache_key]


def _get_image_proc_module(context: InvocationContext) -> ModuleType | None:
    global _IMAGE_PROC_MODULE

    if _IMAGE_PROC_MODULE is not None:
        return _IMAGE_PROC_MODULE

    cache_dir = Path.home() / ".cache" / "birefnet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_proc_path = cache_dir / "image_proc.py"

    if not image_proc_path.exists():
        url = "https://raw.githubusercontent.com/ZhengPeng7/BiRefNet/main/image_proc.py"
        try:
            urllib.request.urlretrieve(url, image_proc_path)
            context.logger.info("Downloaded image_proc.py from BiRefNet repository")
        except Exception as e:
            context.logger.warning(f"Failed to download image_proc.py: {e}")
            return None

    try:
        spec = importlib.util.spec_from_file_location("image_proc", image_proc_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _IMAGE_PROC_MODULE = module
        return module
    except Exception as e:
        context.logger.warning(f"Failed to import image_proc module: {e}")
        return None


def _preprocess(image: Image.Image, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)
    if device.type == "cuda":
        tensor = tensor.half()

    return tensor


def _predict_mask(
    image: Image.Image,
    model_name: str,
    device: torch.device,
) -> Image.Image:
    model = _get_model(model_name, device)
    input_tensor = _preprocess(image, device)

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    mask = preds[0].squeeze()
    mask_pil = transforms.ToPILImage()(mask)
    return mask_pil.resize(image.size, Image.BILINEAR)


@invocation(
    "birefnet",
    title="Remove Background (BiRefNet)",
    tags=["background", "removal", "mask", "birefnet"],
    category="image",
    version="0.0.2",
)
class BiRefNetInvocation(BaseInvocation):
    """Remove image background using BiRefNet."""

    image: ImageField = InputField(description="Input image")
    model: ModelType = InputField(default="BiRefNet", description="Model variant")
    refine_foreground: bool = InputField(
        default=False,
        description="Apply foreground color estimation to prevent background bleed"
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        device = TorchDevice.choose_torch_device()

        mask = _predict_mask(image, self.model, device)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.refine_foreground:
            image_proc = _get_image_proc_module(context)
            if image_proc is not None:
                device_str = "cuda" if device.type == "cuda" else "cpu"
                refined_fg = image_proc.refine_foreground(image, mask, device=device_str)
                r, g, b = refined_fg.split()
            else:
                context.logger.warning("refine_foreground unavailable, using original image")
                r, g, b = image.split()
        else:
            r, g, b = image.split()

        rgba = Image.merge("RGBA", (r, g, b, mask))

        image_dto = context.images.save(image=rgba)
        return ImageOutput.build(image_dto)
