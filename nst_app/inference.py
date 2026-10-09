from pathlib import Path

import torch
from torchvision import transforms

from nst_app.ml.models import Decoder, VGGEncoder
from nst_app.ml.utils import adaptive_instance_normalization


class StyleTransferService:
    def __init__(self, vgg_path, decoder_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = VGGEncoder(vgg_path).to(self.device)
        self.decoder = Decoder().to(self.device)
        self.decoder.load_state_dict(
            torch.load(decoder_path, map_location=self.device)
        )
        self.encoder.eval()
        self.decoder.eval()
        self.transform = transforms.Compose(
            [transforms.Resize(512), transforms.ToTensor()]
        )

    def stylize(self, content_image, style_image, alpha):
        content = self.transform(content_image).unsqueeze(0).to(self.device)
        style = self.transform(style_image).unsqueeze(0).to(self.device)
        return self._stylize_tensors(content, style, alpha)

    def stylize_prepared(self, content_image, style_image, alpha=1.0):
        """Use the comparison's shared resolution without another resize."""
        content = transforms.ToTensor()(content_image).unsqueeze(0).to(self.device)
        style = transforms.ToTensor()(style_image).unsqueeze(0).to(self.device)
        return self._stylize_tensors(content, style, alpha)

    def _stylize_tensors(self, content, style, alpha):
        with torch.no_grad():
            content_features = self.encoder(content, is_test=True)
            style_features = self.encoder(style, is_test=True)
            target = adaptive_instance_normalization(
                content_features, style_features
            )
            target = alpha * target + (1 - alpha) * content_features
            return self.decoder(target)

    @staticmethod
    def save_image(image, path):
        output = image.detach().cpu().squeeze(0).clamp(0, 1)
        transforms.ToPILImage()(output).save(Path(path))
