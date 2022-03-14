# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
"""Make all the CSV files to be turned into figures.

TODO replace everything below (copied from dctests.py)
"""
import logging
import os
import yaml
from typing import List
import sys

sys.path.append("..")
from common.libs import pt_helpers
from compression.libs import initfun
from compression.libs import model_ops
from compression.models import standard_compressor
from common.libs import utilities
from dc.libs import dc_common_args
from dc.tools import dctests

RESULTS_ROOT_DPATH = os.path.join("..", "..", "results", "dc")
logger = logging.getLogger("ImageCompression")


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
    # parser.add_argument(
    #     "--testfun_as_train",
    #     action="store_true",
    #     help="Apply the tests that are normally part of the DC training loop.",
    # )
    parser.add_argument(
        "--list_of_models_yamlpath",
        help=(
            ".yaml file containing a list of models to test with at least the "
            "following keys: models_rootdir, model_name, train_method, train_lambda"
        ),
        default=os.path.join("config", "models_to_test.yaml"),
    )
    parser.add_argument(
        "--list_of_tests_yamlpath",
        help=(
            ".yaml file containing a list of tests: xaxis and yaxis taken from "
            "the model results, with tests performed as necessary"
        ),
        default=os.path.join("config", "test_comparisons.yaml"),
    )
    parser.add_argument(
        "--overwrite_train_res",
        action="store_true",
        help=(
            "Overwrite results present in a model's train_res.json (individual "
            "image results will still be used if available.)"
        ),
    )
    parser.add_argument(
        "--overwrite_individual_img_res",
        action="store_true",
        help="Overwrite individual image results.",
    )


def parser_autocomplete(args):
    """Autocomplete args after they've been parsed."""
    # if args.pretrain_prefix is None:
    #    args.pretrain_prefix = "val" if args.val_dpath is not None else "test"
    check_parameters(args)


def check_parameters(args):
    """Various assertions on parameters/args."""
    assert os.path.isfile(args.list_of_models_yamlpath)


# def testfun_as_train(model, config, jsonsaver, step, device):
#     """
#     Perform the same tests that would be done in the training loop and update
#     the json results file.
#     """
#     _, test_std_loaders = datasets.get_val_test_loaders(None, config["test_std_dpaths"])
#     test_denoise_loader = train.get_test_denoise_loader(config)
#     loss_cls = pt_helpers.get_lossclass(config["lossf"]).to(device)
#     res = train.dctrain_tests(
#         model,
#         step,
#         config,
#         jsonsaver,
#         test_denoise_loader,
#         test_std_loaders,
#         device=device,
#         loss_cls=loss_cls,
#         tb_logger=None,
#     )
#     print(f"Results (visual_loss, bpp):")
#     for reskey, resval in res.items():
#         print(f"{reskey}: {resval}")


# def testfun_JDC-UDd(model, config, jsonsaver, step, device):
#     pass
#     #  TODO make a pre-denoise test set


def expand_models(models: List[dict]) -> list:
    """
    Take list of models parsed from yaml file and return list expanded to cover
    all quality settings (wrt traditional codecs).
    """
    newlist = list()
    for amodel in models:
        if amodel["model_name"] not in (
            "bpg",
            "jpg",
            "jxl",
        ):  # amodel["train_method"] is not None:
            newlist.append(amodel)
            continue
        stdclass = standard_compressor.classes_dict[amodel["model_name"]]
        for val in range(*stdclass.QUALITY_RANGE):
            newmodel = amodel.copy()
            newmodel["train_lambda"] = val
            newmodel["model_name"] += f"_{val}"
            newlist.append(newmodel)
    return newlist


def cleanup_dicts_keys(list_of_dicts: List[dict], bad_prefixes: list) -> List[dict]:
    newlist = []
    for adict in list_of_dicts:
        cdict = adict.copy()
        newlist.append(cdict)
        for bad_prefix in bad_prefixes:
            for akey in list(cdict.keys()):
                if bad_prefix in akey:
                    cdict[akey.removeprefix(bad_prefix)] = cdict.pop(akey)
    return newlist


if __name__ == "__main__":
    """
    FIXME getting initfun arguments but none of training.
    Should get relevant args.
    Maybe initfun could get custom common args function
    ??? (inherited from dctests.py)
    """
    args = initfun.get_args(
        [dc_common_args.parser_add_arguments, parser_add_arguments],
        parser_autocomplete,
        def_config_fpaths=dc_common_args.DC_DEFCONF_FPATHS,
    )
    # jsonsaver = initfun.get_jsonsaver(args)
    device = pt_helpers.get_device(args.device)
    check_parameters(args)
    with open(args.list_of_models_yamlpath, "r") as fp:
        models: List[dict] = yaml.safe_load(fp)
    models = expand_models(models)
    with open(args.list_of_tests_yamlpath, "r") as fp:
        tests = yaml.safe_load(fp)
    for atest in tests:
        results = []

        for model_dict in models:
            prefix = atest["prefix"]
            args.quality_whole_csv_fpath = model_dict.get(
                "quality_whole_csv_fpath", "../../datasets/NIND-msssim.csv"
            )
            print(model_dict)
            if model_dict["train_method"] == "twomodels":
                prefix = model_dict["train_method"] + "_" + prefix
                # args.test_dir = os.path.join()

            xaxis = f'{prefix}_{atest["xaxis"]}'
            yaxis = f'{prefix}_{atest["yaxis"]}'

            res = model_dict.copy()
            # Set parameters of model to be loaded
            args.checkpoints_dpath = os.path.join(
                "..", "..", "models", model_dict.get("models_rootdir", "dc")
            )
            args.expname = model_dict["model_name"]
            args.save_path = os.path.join(args.checkpoints_dpath, args.expname)
            args.train_lambda = model_dict["train_lambda"]
            args.lossf = model_dict.get("lossf", "mse")
            if (
                (not model_dict["train_method"])
                or model_dict["model_name"][:3] in ("bpg", "jpg", "jxl")
                or model_dict.get("arch") == "Passthrough"
            ):
                # if "models_rootdir" not in model_dict:
                args.pretrain = None
                args.arch = model_dict["arch"]
                args.quality = args.train_lambda
            elif model_dict["models_rootdir"] == "dc":
                # args.pretrain_prefix = "val_denoise"
                args.pretrain_prefix = "val_denoise"
                args.passthrough_ae = True
                args.pretrain = model_dict.get("pretrain", args.expname)
                args.arch = model_dict.get("arch", "ManyPriors")
            elif model_dict["models_rootdir"] == "compression":
                # args.pretrain_prefix = "test"
                args.pretrain_prefix = "test"
                args.passthrough_ae = False
                args.pretrain = model_dict.get("pretrain", args.expname)
                args.arch = model_dict.get("arch", "ManyPriors")

            # Load model
            print(args)
            global_step, model = model_ops.get_step_and_loaded_model(
                config=vars(args), device=device
            )
            # load jsonsaver
            jsonsaver = initfun.get_jsonsaver(args)
            # print(jsonsaver.results)
            if (
                (not jsonsaver.is_empty())
                and yaxis in jsonsaver.results[global_step]
                and xaxis in jsonsaver.results[global_step]
                and not args.overwrite_train_res
            ):
                res[xaxis] = jsonsaver.results[global_step][xaxis]
                res[yaxis] = jsonsaver.results[global_step][yaxis]
            else:
                dctests.get_results_dispatcher(
                    model=model,
                    config=vars(args),
                    jsonsaver=jsonsaver,
                    step=global_step,
                    device=device,
                    needed_key=yaxis,
                    prefix=prefix,
                    use_cache=not args.overwrite_individual_img_res,
                )
                res[xaxis] = jsonsaver.results[global_step][xaxis]
                res[yaxis] = jsonsaver.results[global_step][yaxis]
            print(res)
            results.append(res)
            args.pretrain = None
        # save csv
        result_fpath = os.path.join(
            RESULTS_ROOT_DPATH, f'{prefix}-{atest["xaxis"]}-{atest["yaxis"]}.csv'
        )
        result_fpath = result_fpath.removeprefix("twomodels_")
        # cleanup dict if it contains twomodels_
        cleanresults = cleanup_dicts_keys(results, bad_prefixes=["twomodels_"])
        utilities.save_listofdict_to_csv(cleanresults, result_fpath, mixed_keys=True)
