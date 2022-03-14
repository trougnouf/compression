"""
Compare image directories with the same structures and save their respective scores to a csv file.

Intended use case is to make std (noisy) - denoised (gt) pairs list
"""

import os
import sys
import configargparse
import piqa

sys.path.append("..")
# pylint: disable=wrong-import-position
from common.libs import utilities
from common.libs import pt_helpers

# from common.libs import libimganalysis

XDPATHS = (
    os.path.join(
        "..",
        "..",
        "datasets",
        "denoised",
        "2021-06-14T20:27_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_with_clean_data.yaml_--g_model_path_..-..-models-nind_denoise-2021-06-12T11:48_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_w",
        "cropped",
        "NIND_256_192",
    ),
    os.path.join(
        "..",
        "..",
        "datasets",
        "train",
        "denoised",
        "2021-06-14T20:27_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_with_clean_data.yaml_--g_model_path_..-..-models-nind_denoise-2021-06-12T11:48_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_w",
        "resized",
        "1024",
        "FeaturedPictures",
    ),
)
YDPATHS = (
    os.path.join("..", "..", "datasets", "cropped", "NIND_256_192"),
    os.path.join(
        "..", "..", "datasets", "train", "resized", "1024", "FeaturedPictures"
    ),
)
DEVICE = 0


def parse_args():
    """Argument parser."""
    parser = configargparse.ArgumentParser(
        description=__doc__,
        default_config_files=[],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument(
        "--xdpaths",
        nargs="+",
        default=XDPATHS,
        help=(
            "Directory/ies containing images to be used as ground-truth."
            "Result will be this+'-msssim.csv'"
        ),
    )
    parser.add_argument(
        "--ydpaths",
        nargs="+",
        default=YDPATHS,
        help="Path of directory/ies containing images to be used as noisy",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        type=int,
        help="Device number (default: 0, typically 0-3)",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()
    device = pt_helpers.get_device(args.device)
    loss = piqa.MS_SSIM().to(device)
    missing_files = list()
    for xdpath, ydpath in zip(args.xdpaths, args.ydpaths):
        output_fpath = xdpath + "-msssim.csv"
        results = list()
        for root, adir, fn in utilities.walk(xdpath):
            xpath = os.path.join(root, adir, fn)
            ypath = os.path.join(ydpath, adir, fn)
            #  assert os.path.isfile(ypath), ypath
            if not os.path.isfile(ypath):
                missing_files.append((xpath, ypath))
                print("compare_imagedirs.py: warning: missing {ypath}")
                continue
            #  score = libimganalysis.piqa_msssim(xpath, ypath)
            score = loss(  # creating a single class to minimize GPU/CPU transfers
                pt_helpers.fpath_to_tensor(xpath, batch=True, device=device),
                pt_helpers.fpath_to_tensor(ypath, batch=True, device=device),
            ).item()
            print(f"{xpath=}, {ypath=}, {score=}")
            results.append((xpath, ypath, score))
        utilities.list_of_tuples_to_csv(
            results, ("xpath", "ypath", "score"), output_fpath
        )
        print(f"Quality check exported to {output_fpath}")
    print(f"Missing files: {missing_files}")
