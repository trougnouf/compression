"""Datasets and their loaders used by compression."""
import os
from glob import glob
from typing import Union, List
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset

# from data_loader.datasets import Dataset
import torch
import sys

sys.path.append("..")
try:
    from siren import dataio
except ModuleNotFoundError:
    print("datasets.py: warning: could not load siren dataset format")
from common.libs import utilities
from common.libs import pt_ops


class Datasets(Dataset):
    """Load and crop image w/ basic data augmentation."""

    def __init__(self, data_dpaths, image_size=256):
        # self.data_dir = data_dir
        if isinstance(data_dpaths, str):
            data_dpaths = [data_dpaths]
        self.image_size = image_size
        self.image_paths = []
        for data_dir in data_dpaths:
            if not os.path.exists(data_dir):
                raise Exception(f"Datasets error: {data_dir} does not exist")
            self.image_paths.extend(sorted(glob(os.path.join(data_dir, "*.*"))))

    def __getitem__(self, item):
        img_fpath = self.image_paths[item]
        image = Image.open(img_fpath).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        return transform(image)

    def __len__(self):
        return len(self.image_paths)


class Datasets_Img_Coords(Datasets):
    """Used by SIREN-like networks (deprecated)."""

    def __init__(self, data_dpaths, image_size=256):
        super().__init__(data_dpaths, image_size)
        assert isinstance(image_size, int)  # TODO test dif shape s.a. 768*512
        self.mgrid = dataio.get_mgrid((image_size, image_size))

    def __getitem__(self, item):
        img = super().__getitem__(item)
        return img


class TestDirDataset(Dataset):
    """Data loader for an image directory (or two)."""

    def __init__(
        self,
        data_dir,
        data_dir_2=None,
        resize=None,
        verbose=False,
        crop_to_multiple=None,
        incl_fpaths=False,
        return_size_of_img2=True,
    ):
        """
        data_dir: contains images
        data_dir_2: contains matching images (returned as the 2nd value, y)
        resize: not not implemented
        verbose
        crop_to_multiple: recommended 64
        incl_fpaths: returns tensor, fn or (tensor, tensor), (fpath, fpath)
        return_size_of_img2: returns tensor, tensor, sizeof(img2)
        """
        self.data_dir = data_dir
        self.data_dir_2 = data_dir_2
        if resize is not None:
            resize = int(resize)
            raise NotImplementedError("resize")
        self.resize = resize
        if not os.path.exists(data_dir):
            raise Exception(f"TestDirDataset error: {self.data_dir} does not exist")
        elif os.path.isdir(data_dir):
            self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))
        elif os.path.isfile(data_dir):
            self.image_path = [data_dir]
        if data_dir_2 is not None:
            if not os.path.exists(data_dir_2):
                raise ValueError(f"data_dir_2={data_dir_2} does not exist")
                # raise ValueError(f'{data_dir_2=} does not exist')  # FIXME restore this (req modern python3)
            self.image_path_2 = sorted(
                list(
                    filter(
                        lambda fpath: not fpath.endswith("txt"),
                        glob(os.path.join(data_dir_2, "*.*")),
                    )
                )
            )
            # self.image_path_2 = sorted(glob(os.path.join(data_dir_2, "*.png")))
            # if len(self.image_path_2) == 0:
            #     self.image_path_2 = sorted(glob(os.path.join(data_dir_2, "*.jpg")))
            # if len(self.image_path_2) == 0:
            #     self.image_path_2 = sorted(glob(os.path.join(data_dir_2, "*.tif")))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.verbose = verbose
        self.crop_to_multiple = crop_to_multiple
        self.incl_fpaths = incl_fpaths
        self.return_size_of_img2 = return_size_of_img2

    def __getitem__(self, item):
        img_fpath = self.image_path[item]
        if self.verbose:
            print("{}: loading {}".format(self, img_fpath))
        image = Image.open(img_fpath).convert("RGB")
        image = self.transform(image)
        ch, h, w = image.shape
        # image = image[:, :h-h%64, :w-w%64] # ensure divisible by 16, actually no longer necessary bc taken care of in preprocessing
        if hasattr(self, "image_path_2"):
            img2_fpath = self.image_path_2[item]
            assert utilities.get_leaf(img2_fpath).split(".")[0] in img_fpath, (
                img_fpath,
                img2_fpath,
            )
            image2 = Image.open(img2_fpath).convert("RGB")
            image2 = self.transform(image2)

            if self.crop_to_multiple is not None:
                result = (
                    pt_ops.crop_to_multiple(image, self.crop_to_multiple),
                    pt_ops.crop_to_multiple(image2, self.crop_to_multiple),
                )
            else:
                result = (image, image2)
            if self.return_size_of_img2:
                # print(self.image_path_2)
                # print(item)

                # TODO what is the use_case? need to make size an option
                if os.path.exists(img2_fpath + ".bpg"):
                    size = utilities.filesize(img2_fpath + ".bpg")
                if os.path.exists(img2_fpath + ".jxl"):
                    size = utilities.filesize(img2_fpath + ".jxl")
                else:
                    size = utilities.filesize(img2_fpath)
                result = result, size
            if self.incl_fpaths:
                result = result, (img_fpath, img2_fpath)
        else:
            if self.crop_to_multiple is not None:
                result = pt_ops.crop_to_multiple(image, self.crop_to_multiple)
            else:
                result = image
            if self.incl_fpaths:
                result = result, img_fpath
        return result

    def __len__(self):
        return len(self.image_path)


TestKodakDataset = TestDirDataset


def get_val_test_loaders(
    val_dpath, test_dpaths: Union[str, List[str]], artificial_noise=False
):
    """returns a validation loader and a dictionary of test loaders."""
    test_loaders = dict()
    if test_dpaths is not None:
        if isinstance(test_dpaths, str):
            test_dpaths = [test_dpaths]
        for test_dpath in test_dpaths:
            test_dataset = TestDirDataset(data_dir=test_dpath)
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                shuffle=False,
                batch_size=1,
                pin_memory=True,
                num_workers=1,
            )
            test_loaders[utilities.get_leaf(test_dpath)] = test_loader
    if val_dpath is None:
        val_loader = None
    else:
        val_dataset = TestDirDataset(data_dir=val_dpath)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            num_workers=1,
        )
    return val_loader, test_loaders
