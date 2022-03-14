# -*- coding: utf-8 -*-
import math
from typing import List, Optional
import sys
import torch

sys.path.append("..")
from compression.libs import datasets
from nind_denoise import dataset_torch_3


class StdDataset(datasets.Datasets):
    """
    Load and crop image w/ basic data augmentation. Return x,y

    This extension allows for artificial noise. Values are those used by
    Testolina et al. (2021).
    """

    def __init__(
        self,
        data_dpaths: List[str],
        image_size: int = 256,
        add_artificial_noise: bool = False,
        poisson_noise_component: Optional[float] = 0.2 ** 2,
        gaussian_noise_component: Optional[float] = 0.04 ** 2,
    ):
        super().__init__(data_dpaths=data_dpaths, image_size=image_size)
        self.artificial_noise_wanted = add_artificial_noise
        if self.artificial_noise_wanted:
            assert (
                poisson_noise_component is not None
                and gaussian_noise_component is not None
            )
            self.poisson_noise_component = poisson_noise_component
            self.gaussian_noise_component = gaussian_noise_component

    def __getitem__(self, i):
        pt_image_batch = super().__getitem__(i)
        if self.artificial_noise_wanted:
            return pt_image_batch, add_artificial_noise(
                pt_image_batch,
                poisson_noise_component=self.poisson_noise_component,
                gaussian_noise_component=self.gaussian_noise_component,
            )
        else:
            return pt_image_batch, pt_image_batch


class ArtificialNoiseNIND(dataset_torch_3.DenoisingDataset):
    def __init__(
        self,
        datadirs: List[str],
        test_reserve: list = [],
        min_crop_size: Optional[int] = None,
        exact_reserve: bool = False,
        cs=None,
        poisson_noise_component: Optional[float] = 0.2 ** 2,
        gaussian_noise_component: Optional[float] = 0.04 ** 2,
        **kwargs
    ):
        super().__init__(
            datadirs=datadirs,
            test_reserve=test_reserve,
            min_crop_size=min_crop_size,
            exact_reserve=exact_reserve,
            cs=cs,
            **kwargs,
        )
        self.poisson_noise_component = poisson_noise_component
        self.gaussian_noise_component = gaussian_noise_component

    def __getitem__(self, i):
        x_image_batch = super().__getitem__(i)[0]
        return x_image_batch, add_artificial_noise(
            x_image_batch,
            poisson_noise_component=self.poisson_noise_component,
            gaussian_noise_component=self.gaussian_noise_component,
        )


def add_artificial_noise(
    pt_image_batch,
    poisson_noise_component: float,
    gaussian_noise_component: float,
):
    """
    Adapted from demo_ClipPoisGaus_stdEst2D.m, L172.

    https://webpages.tuni.fi/foi/sensornoise.html
    """
    if poisson_noise_component == 0:
        result = pt_image_batch
    else:
        chi = 1 / poisson_noise_component
        result = torch.poisson(torch.clamp_min(chi * pt_image_batch, 0)) / chi
    result = result + math.sqrt(gaussian_noise_component) * torch.rand_like(
        pt_image_batch
    )
    result = torch.clip(result, 0, 1)
    return result
