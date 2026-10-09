import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from nst_app import PROJECT_ROOT, create_app
from nst_app.comparison import FastStyleTransferService, prepare_image


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.uploads = tempfile.TemporaryDirectory()
        cls.app = create_app({
            "TESTING": True, "UPLOAD_FOLDER": cls.uploads.name,
            "COMPARISON_IMAGE_SIZE": 256,
        })

    @classmethod
    def tearDownClass(cls):
        cls.uploads.cleanup()

    def setUp(self):
        self.client = self.app.test_client()

    def token(self):
        response = self.client.get("/compare")
        self.assertEqual(response.status_code, 200)
        return re.search(rb'name="compare-csrf_token"[^>]*value="([^"]+)"', response.data)[1].decode()

    def test_real_checkpoint_and_both_methods_with_fixed_style(self):
        token = self.token()
        comparison = self.app.extensions["comparison"]
        original = comparison.adain.stylize_prepared
        with patch.object(comparison.adain, "stylize_prepared", wraps=original) as adain:
            with (PROJECT_ROOT / "assets/examples/brad_pitt.jpg").open("rb") as image:
                response = self.client.post("/compare", data={
                    "compare-csrf_token": token,
                    "compare-content": (image, "content.jpg"),
                    # Forged style/alpha must not influence the fixed comparison.
                    "style": (io.BytesIO(b"ignore"), "other.jpg"), "alpha": "0",
                })
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"We couldn&#39;t compare", response.data)
        self.assertIn(b"Both outputs: 256", response.data)
        self.assertEqual(adain.call_args.args[2], 1.0)
        with Image.open(comparison.style_path) as reference:
            expected = prepare_image(reference.convert("RGB"), 256)
        self.assertEqual(adain.call_args.args[1].tobytes(), expected.tobytes())
        names = re.findall(rb'href="/uploads/(compare_[^"]+\.png)" download', response.data)
        self.assertEqual(len(names), 2)
        for name in names:
            downloaded = self.client.get("/uploads/" + name.decode())
            self.assertEqual(downloaded.status_code, 200)
            with Image.open(io.BytesIO(downloaded.data)) as result:
                self.assertEqual(result.size, (256, 256))
                self.assertEqual(result.mode, "RGB")
                self.assertTrue(any(low != high for low, high in result.getextrema()))
            downloaded.close()

    def test_invalid_and_missing_uploads_do_not_run_models(self):
        comparison = self.app.extensions["comparison"]
        with patch.object(comparison, "compare") as run:
            for upload, message in (
                (None, b"Please choose a content image"),
                ((io.BytesIO(b"file"), "content.txt"), b"Images must be PNG or JPEG"),
                ((io.BytesIO(b"not an image"), "content.jpg"), b"We couldn&#39;t compare"),
            ):
                data = {"compare-csrf_token": self.token()}
                if upload:
                    data["compare-content"] = upload
                response = self.client.post("/compare", data=data)
                self.assertEqual(response.status_code, 200)
                self.assertIn(message, response.data)
            run.assert_not_called()

    def test_csrf_is_required(self):
        with patch.object(self.app.extensions["comparison"], "compare") as run:
            response = self.client.post("/compare", data={
                "compare-content": (io.BytesIO(b"file"), "content.jpg"),
            })
            self.assertIn(b"CSRF token is missing", response.data)
            run.assert_not_called()

    def test_rectangle_keeps_shared_dimensions_and_small_images_rejected(self):
        image = prepare_image(Image.new("RGB", (701, 403)), 256)
        self.assertEqual(image.size, (256, 144))
        with self.assertRaises(ValueError):
            prepare_image(Image.new("RGB", (1024, 8)), 256)
        fast = FastStyleTransferService(PROJECT_ROOT / "models/mosaic.pth", torch.device("cpu"))
        output = fast.stylize(Image.new("RGB", (64, 40), (200, 120, 60)))
        self.assertEqual(tuple(output.shape), (1, 3, 40, 64))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreaterEqual(output.min().item(), 0)
        self.assertLessEqual(output.max().item(), 1)

    def test_existing_playground_and_style_reference(self):
        response = self.client.get("/")
        self.assertIn(b'action="/"', response.data)
        self.assertIn(b'name="style"', response.data)
        reference = self.client.get("/comparison-style")
        self.assertEqual(reference.status_code, 200)
        reference.close()


if __name__ == "__main__":
    unittest.main()
