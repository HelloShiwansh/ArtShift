# Pretrained Mosaic Fast NST

Architecture and pretrained weights come from the official PyTorch examples:
https://github.com/pytorch/examples/tree/main/fast_neural_style

- Checkpoint: `mosaic.pth`, extracted from the official `saved_models.zip`.
- Archive URL: https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1
- Matching reference: https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/images/style-images/mosaic.jpg
- Architecture source: https://github.com/pytorch/examples/blob/main/fast_neural_style/neural_style/transformer_net.py
- License: `FAST_NST_LICENSE.txt` (BSD 3-Clause).

This is the Johnson perceptual-loss method with instance normalization. No
training is performed in this project for this checkpoint. The model accepts RGB
pixels in [0, 255]; its output is converted to [0, 1] before using the application's
image-saving helper. Deprecated InstanceNorm running-mean/variance keys are
removed when loading the official older checkpoint.

The comparison uses the matching Mosaic image as AdaIN's fixed style reference,
with alpha=1. Both methods receive exactly the same prepared content image.
