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
