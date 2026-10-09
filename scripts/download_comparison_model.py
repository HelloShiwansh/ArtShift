"""Restore the official Mosaic checkpoint and matching reference image."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_URL = "https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1"
STYLE_URL = "https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/mosaic.jpg"


def main():
    with urlopen(ARCHIVE_URL, timeout=120) as response:
        with ZipFile(BytesIO(response.read())) as archive:
            checkpoint = archive.read("saved_models/mosaic.pth")
    with urlopen(STYLE_URL, timeout=60) as response:
        reference = response.read()
    for path, data in (
        (ROOT / "models/mosaic.pth", checkpoint),
        (ROOT / "assets/examples/mosaic.jpg", reference),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"Saved {path.name} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
