# pylint: disable=wrong-import-position
"""Tests specific to denoising-compression."""
import time
import logging
import os
import statistics
import math
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

sys.path.append("..")
from common.libs import pt_helpers
from common.extlibs import pt_ms_ssim
from compression.libs import datasets
from compression.libs import initfun
from compression.libs import model_ops
from common.libs import utilities
from common.libs import pt_ops
from compression.libs import datasets
from dc import train
from dc.libs import dc_common_args
from compression import tests


logger = logging.getLogger("ImageCompression")

DENOISER = "2021-06-14T20:27_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_with_clean_data.yaml_--g_model_path_..-..-models-nind_denoise-2021-06-12T11:48_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_w"


def parser_add_arguments(parser) -> None:
    """List and parser of arguments specific to this part of the project."""
    # These options are required (but already defined in train.py)
    # parser.add_argument(
    #     "--expname",
    #     type=str,
    #     help="Experiment name to load",
    # )
    # parser.add_argument(
    #     "--epoch",
    #     type=int,
    #     help="Decode a given bitstream. Output will be save_path/decoded/arg+.png if not specified",
    # )
    # parser.add_argument(
    #     "--arch",
    #     action="store_true",
    #     help="Network architectur",
    # )
    parser.add_argument(
        "--testfun_as_train",
        action="store_true",
        help="Apply the tests that are normally part of the DC training loop.",
    )


def parser_autocomplete(args):
    """Autocomplete args after they've been parsed."""
    if args.pretrain_prefix is None:
        args.pretrain_prefix = "val" if args.val_dpath is not None else "test"
    check_parameters(args)


def check_parameters(args):
    """Various assertions on parameters/args."""
    pass


def get_results_dispatcher(
    model,
    config,
    jsonsaver,
    step,
    device,
    needed_key,
    prefix,
    use_cache=True,
):
    config["max_quality"] = None
    if "_q>=" in needed_key:
        config["min_quality"] = float(needed_key.split("q>=")[1].split(";")[0])
        config["max_quality"] = 1.0
    elif "_q=[" in needed_key:
        config["min_quality"] = float(needed_key.split("q=[")[1].split(",")[0])
        config["max_quality"] = float(
            needed_key.split("q=[")[1].split(",")[1].split("]")[0]
        )
    # elif prefix == "test_clic_pro" or prefix == "kodak":
    #     pass
    # prefix = needed_key.split(";")[0]
    if "test_denoise_" in needed_key and "_psnr" in needed_key:
        get_test_denoise_bpp_mse(
            model,
            config,
            jsonsaver,
            step,
            device,
            prefix=prefix,
            use_cache=use_cache,
        )
    elif "test_denoise_" in needed_key and "_msssim" in needed_key:
        get_test_denoise_bpp_msssim(
            model,
            config,
            jsonsaver,
            step,
            device,
            prefix=prefix,
            use_cache=use_cache,
        )
    # elif "test_clic_pro_" in needed_key:
    #     get_test_dir(model, config, jsonsaver, step, device, prefix=prefix)
    # elif "kodak" in needed_key:
    #     get_test_dir(model, config, jsonsaver, step, device, prefix=prefix)
    else:  # regular images
        # raise NotImplementedError(needed_key)
        get_test_dir(
            model,
            config,
            jsonsaver,
            step,
            device,
            prefix=prefix,
            use_cache=use_cache,
        )


def get_test_denoise_bpp(
    model, config, jsonsaver, step, device, prefix, loss_str, use_cache=True
):
    if "twomodels" in prefix:
        config["ground_truth_y_dpath"] = os.path.join(
            "..", "..", "datasets", "denoised", DENOISER, "NIND"
        )
    else:
        config["ground_truth_y_dpath"] = None
    test_denoise_loader = train.get_test_denoise_loader(config, incl_fpaths=True)
    loss_cls = pt_helpers.get_lossclass(loss_str).to(device)
    # cache_prefix = step if not hasattr(model, "quality") else model.quality
    individual_results_cache_fpath = os.path.join(
        config["save_path"],
        "tests",
        str(step) + "_individual_results_cache.yaml",
    )
    if os.path.isfile(individual_results_cache_fpath):
        with open(individual_results_cache_fpath, "r") as fp:
            individual_results_cache = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        individual_results_cache = dict()
    tests.test_dir(
        model=model,
        step=step,
        jsonsaver=jsonsaver,
        config=config,
        device="cpu" if config["test_on_cpu"] else device,
        prefix=prefix,
        loader=test_denoise_loader,
        tb_logger=None,
        loss_cls=loss_cls,
        gt_src="tuple_fpaths",
        incl_combined_loss=True,
        individual_results_cache=individual_results_cache,
        use_cache=use_cache,
    )
    with open(individual_results_cache_fpath, "w") as fp:
        yaml.dump(individual_results_cache, fp)


def get_test_denoise_bpp_mse(model, config, jsonsaver, step, device, prefix, use_cache):
    return get_test_denoise_bpp(
        model, config, jsonsaver, step, device, prefix, "mse", use_cache
    )


def get_test_denoise_bpp_msssim(
    model, config, jsonsaver, step, device, prefix, use_cache
):
    return get_test_denoise_bpp(
        model, config, jsonsaver, step, device, prefix, "msssim", use_cache
    )


def get_test_dir(model, config, jsonsaver, step, device, prefix, use_cache=True):
    data_dir_2 = None
    prefix_ = prefix
    if "twomodels" in prefix:
        prefix_ = prefix.split("twomodels_")[-1]
        data_dir_2 = os.path.join(
            "..", "..", "datasets", "test", "denoised", DENOISER, prefix_
        )
    try:
        test_dataset = datasets.TestDirDataset(
            data_dir=os.path.join("..", "..", "datasets", "test", prefix_),
            data_dir_2=data_dir_2,
            incl_fpaths=True,
            return_size_of_img2=False,
        )
    except ValueError as e:
        utilities.popup(f"dctests.get_test_dir error {e=}. Does {data_dir_2=} exist?")
        breakpoint()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
    )
    loss_cls = pt_helpers.get_lossclass(config.get("lossf", "mse")).to(device)
    # cache_prefix = step if not hasattr(model, "quality") else model.quality
    individual_results_cache_fpath = os.path.join(
        config["save_path"],
        "tests",
        str(step) + "_individual_results_cache.yaml",
    )
    if os.path.isfile(individual_results_cache_fpath):
        with open(individual_results_cache_fpath, "r") as fp:
            individual_results_cache = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        individual_results_cache = dict()
    try:
        tests.test_dir(
            model=model,
            step=step,
            jsonsaver=jsonsaver,
            config=config,
            device="cpu" if config["test_on_cpu"] else device,
            prefix=prefix,
            loader=test_loader,
            tb_logger=None,
            loss_cls=loss_cls,
            gt_src="tuple_fpaths" if "twomodels" in prefix else "single_wfpath",
            incl_combined_loss=True,
            individual_results_cache=individual_results_cache,
            crop_to_multiple_size=64,
        )
    except ValueError as e:
        utilities.popup(f"dctests.get_test_dir error {e}")
        breakpoint()
    with open(individual_results_cache_fpath, "w") as fp:
        yaml.dump(individual_results_cache, fp)


def testfun_as_train(model, config, jsonsaver, step, device):
    """
    Perform the same tests that would be done in the training loop and update
    the json results file.
    """
    _, test_std_loaders = datasets.get_val_test_loaders(None, config["test_std_dpaths"])
    test_denoise_loader = train.get_test_denoise_loader(config)
    loss_cls = pt_helpers.get_lossclass(config["lossf"]).to(device)
    res = train.dctrain_tests(
        model,
        step,
        config,
        jsonsaver,
        test_denoise_loader,
        test_std_loaders,
        device=device,
        loss_cls=loss_cls,
        tb_logger=None,
    )
    print(f"Results (visual_loss, bpp):")
    for reskey, resval in res.items():
        print(f"{reskey}: {resval}")


# def testfun_predenoised(model, config, jsonsaver, step, device):
#     pass
#     #  TODO make a pre-denoise test set


if __name__ == "__main__":
    """
    FIXME getting initfun arguments but none of training.
    Should get relevant args.
    Maybe initfun could get custom common args function
    """
    args = initfun.get_args(
        [dc_common_args.parser_add_arguments, parser_add_arguments],
        parser_autocomplete,
        def_config_fpaths=dc_common_args.DC_DEFCONF_FPATHS,
    )
    jsonsaver = initfun.get_jsonsaver(args)
    device = pt_helpers.get_device(args.device)
    check_parameters(args)
    global_step, model = model_ops.get_step_and_loaded_model(
        config=vars(args), device=device
    )
    for argkey, argval in vars(args).items():
        if argkey.startswith("testfun_") and argval:
            print(f'Launching test "{argkey}"')
            try:
                locals()[argkey](
                    model=model,
                    config=vars(args),
                    jsonsaver=jsonsaver,
                    step=global_step,
                    device=device,
                )
            except KeyError as e:
                if argkey == e:
                    print(
                        f"dctests.py KeyError ({e}), possibly because no test named {argkey}"
                    )
                else:
                    raise KeyError(e)
