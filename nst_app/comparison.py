"""Compare arbitrary-style AdaIN with a fixed-style pretrained TransformerNet."""
import re
from pathlib import Path
from threading import Lock
from time import perf_counter

import torch
from PIL import Image
from torchvision import transforms

from nst_app.ml.transformer_net import TransformerNet


def prepare_image(image, max_size):
    """Preserve aspect ratio and align dimensions to AdaIN's downsampling factor."""
    width, height = image.size
    scale = min(1.0, max_size / max(width, height))
    size = tuple(max(8, round(dimension * scale / 8) * 8) for dimension in (width, height))
    if min(size) < 16:
        raise ValueError("Please use an image with a less extreme aspect ratio and dimensions of at least 16 pixels.")
    return image.resize(size, Image.Resampling.LANCZOS)


class FastStyleTransferService:
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.model = TransformerNet()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Official older checkpoints contain InstanceNorm buffers no longer used.
        state = {
            key: value for key, value in state.items()
            if not re.search(r"in\d+\.running_(mean|var)$", key)
        }
        self.model.load_state_dict(state)
        self.model.to(device).eval().requires_grad_(False)

    def stylize(self, content_image):
        # These weights expect RGB pixels in [0, 255], unlike AdaIN's [0, 1].
        content = transforms.ToTensor()(content_image).unsqueeze(0).to(self.device) * 255
        with torch.inference_mode():
            return (self.model(content) / 255).clamp(0, 1)


class ComparisonService:
    def __init__(self, adain, checkpoint_path, style_path, image_size=512):
        self.adain = adain
        self.checkpoint_path = Path(checkpoint_path)
        self.style_path = Path(style_path)
        self.image_size = image_size
        self._fast = None
        self._load_lock = Lock()

    def _load_fast(self):
        with self._load_lock:
            if self._fast is None:
                if not self.checkpoint_path.is_file():
                    raise ValueError("The Mosaic model is missing. Please run the comparison setup described in the README.")
                self._fast = FastStyleTransferService(self.checkpoint_path, self.adain.device)
        return self._fast

    def _synchronize(self):
        if self.adain.device.type == "cuda":
            torch.cuda.synchronize(self.adain.device)

    def compare(self, content_image):
        fast = self._load_fast()
        content = prepare_image(content_image.convert("RGB"), self.image_size)
        with Image.open(self.style_path) as reference:
            style = prepare_image(reference.convert("RGB"), self.image_size)
        results = {}
        for name, operation in (
            ("adain", lambda: self.adain.stylize_prepared(content, style, 1.0)),
            ("fast", lambda: fast.stylize(content)),
        ):
            self._synchronize()
            start = perf_counter()
            output = operation()
            self._synchronize()
            results[name] = {"image": output, "seconds": perf_counter() - start}
        results["size"] = content.size
        return results
