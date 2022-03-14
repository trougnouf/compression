"""Passthrough model."""
import sys
import torch

sys.path.append("..")
from compression.models import abstract_model


class Passthrough_ImageCompressor(abstract_model.AbstractImageCompressor):
    """Passthrough model."""

    def __init__(self, device, **kwargs):
        super().__init__(device=device)

    def forward(self, input_image):
        "Returns input image."
        return {"reconstructed_image": input_image, "bpp": torch.zeros(1)}
