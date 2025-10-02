"""BiRefNet background removal for InvokeAI."""

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
    version="0.0.1",
)
class BiRefNetInvocation(BaseInvocation):
    """Remove image background using BiRefNet."""

    image: ImageField = InputField(description="Input image")
    model: ModelType = InputField(default="BiRefNet", description="Model variant")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        device = TorchDevice.choose_torch_device()

        mask = _predict_mask(image, self.model, device)

        if image.mode != "RGB":
            image = image.convert("RGB")

        r, g, b = image.split()
        rgba = Image.merge("RGBA", (r, g, b, mask))

        image_dto = context.images.save(image=rgba)
        return ImageOutput.build(image_dto)
