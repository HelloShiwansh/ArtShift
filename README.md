# AI Neural Style Transfer

A Flask application for arbitrary neural style transfer using AdaIN and PyTorch.

## Project structure

```text
AI-NST-Project/
|-- nst_app/              Flask application package
|   |-- ml/               Neural network models and AdaIN utilities
|   |-- templates/        Web interface
|   |-- forms.py          Upload form
|   |-- inference.py      Style-transfer service
|   `-- web.py            Routes and upload handling
|-- models/               VGG and decoder weights
|-- datasets/             Content and style training images
|-- assets/examples/      Examples displayed by the web app
|-- training/             Training entry point and saved runs
|-- notebooks/            Exploration and feature visualizations
|-- docs/                 Architecture and demo images
|-- instance/uploads/     Local uploads and generated images (ignored)
|-- run.py                Local and production app entry point
`-- requirements.txt      Runtime dependencies
```

## Run locally

The pinned PyTorch version requires Python 3.11.

```powershell
.\.venv\Scripts\activate
python run.py
```

Open http://localhost:5000.

## Compare AdaIN and pretrained Fast NST

Open **Compare** in the navigation. Upload a single PNG or JPEG photograph and
select **Compare methods** to view and download both outputs side by side.

The section fixes the style to **Mosaic**. AdaIN uses the matching reference image
at full style strength (`alpha=1`); Fast NST uses the official pretrained PyTorch
Mosaic TransformerNet checkpoint. Fast NST needs no training and does not accept
an uploaded style image. The existing Playground still supports arbitrary styles.

Both methods receive the same content image, with aspect ratio retained and a
maximum of 512 pixels on the longest edge. Dimensions are rounded to multiples of
8 for AdaIN. Both results are saved as PNGs. Displayed processing times exclude
model loading, shared image resizing, and file saving; they are per-request
measurements, not a controlled performance benchmark. No quality ranking is implied.

The checkpoint (`models/mosaic.pth`) and style (`assets/examples/mosaic.jpg`) are
included. To restore them if missing:

```powershell
python scripts/download_comparison_model.py
```

Source attribution and license are in `models/FAST_NST_SOURCE.md` and
`models/FAST_NST_LICENSE.txt`. Fast NST is loaded on the first comparison and
reused afterward. Runtime uses CUDA if available, otherwise CPU.

Run the comparison checks, including inference with the real pretrained weights:

```powershell
python -m unittest discover -s tests -v
```

## Reinstall dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train the decoder

Run training from the project root so the application package is importable:

```powershell
python -m training.train --experiment my_run
```

Outputs are written to `training/runs/<experiment>`.

## Work with the notebook

Install the optional development dependencies, then open the `notebooks` folder:

```powershell
python -m pip install -r requirements-dev.txt
python -m jupyter lab
```
