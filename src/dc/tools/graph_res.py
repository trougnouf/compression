"""
Graph results generated in ../../results/dc.

to run all:
python tools/graph_res.py --config config/graph_85_onemodel.yaml && python tools/graph_res.py --config config/graph_85_twomodels.yaml && python tools/graph_res.py --config config/graph_lownoise.yaml && python tools/graph_res.py


include:
/orb/benoit_phd/results/dc/graph_lownoise.yaml/twomodels_test_denoise_q=[0.7,0.999];-bpp-msssim.csv.pdf
/orb/benoit_phd/results/dc/graph_85_onemodel.yaml/twomodels_test_denoise_q=[0.7,0.999];-bpp-msssim.csv.pdf
/orb/benoit_phd/results/dc/graph_85_twomodels.yaml/twomodels_test_denoise_q=[0.7,0.999];-bpp-msssim.csv.pdf
"""

import csv
import logging
import os
from typing import Callable, Union, Optional
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")
from common.libs import utilities
from dc.libs import dc_common_args
from compression.libs import initfun

logger = logging.getLogger("ImageCompression")
RES_DPATH = os.path.join("..", "..", "results", "dc")
MARKERSDICT = {
    "std AE": "s",
    "std AE (MS-SSIM)": "8",
    "JDC-Cn.8-Tdec": "$t$",  # "p",
    "JDC-UD": "$P$",  # "8",
    "JDC-Cn.9-MS-SSIM": "*",
    "JDC-Cn.8-MS-SSIM": "*",
}
MARKERSDICT["JDC-Cn.9"] = MARKERSDICT["JDC-Cn.5"] = MARKERSDICT[
    "JDC-Cn.6"
] = MARKERSDICT["JDC-Cn.7"] = MARKERSDICT["JDC-Cn.8"] = MARKERSDICT[
    "JDC-Cn.4"
] = MARKERSDICT[
    "JDC-Cn.2"
] = "o"
MARKERSDICT["JDC-Cn.9"] = "$9$"
MARKERSDICT["JDC-Cn.5"] = "$5$"
MARKERSDICT["JDC-Cn.6"] = "$6$"
MARKERSDICT["JDC-Cn.7"] = "$7$"
MARKERSDICT["JDC-Cn.8"] = "$8$"
MARKERSDICT["JDC-Cn.4"] = "$4$"
MARKERSDICT["JDC-Cn.2"] = "$2$"
MARKERSDICT["JDC-CN"] = "$0$"
MARKERSDICT["JDC-N"] = "$N$"
MARKERSDICT["Testolina"] = "$T$"

MARKERSDICT["JPEG"] = MARKERSDICT["BPG"] = MARKERSDICT["JPEGXL"] = ","#"s"
MARKERSDICT[
    "std AE"
] = "s"
MARKERSDICT["denoise+JPEG"] = MARKERSDICT["denoise+BPG"] = MARKERSDICT[
    "denoise+JPEGXL"
] = ","# "$D$"
MARKERSDICT["denoise+std AE"] = "$D$"
MARKERSDICT["denoise+std AE (MS-SSIM)"] = "$d$"
COLORSDICT = {
    "JDC-N": "violet",
    "JDC-CN": "red",
    "JDC-Cn.2": "lightsteelblue",
    "JDC-Cn.4": "cornflowerblue",
    "JDC-Cn.5": "indigo",
    "JDC-Cn.6": "purple",
    "JDC-Cn.7": "darkblue",
    "JDC-Cn.8": "blue",
    # "lownoise (85)": "teal",
    "JDC-Cn.9": "cyan",
    "JDC-UD": "orange",
    "JDC-Cn.9-MS-SSIM": "lightblue",
    "JDC-Cn.8-MS-SSIM": "lightblue",
    "JDC-Cn.8-Tdec": "pink",
}
COLORSDICT["std AE"] = COLORSDICT["denoise+std AE"] = COLORSDICT[
    "std AE (MS-SSIM)"
] = COLORSDICT["denoise+std AE (MS-SSIM)"] = "olive"
COLORSDICT["JPEG"] = COLORSDICT["denoise+JPEG"] = "grey"
COLORSDICT["JPEGXL"] = COLORSDICT["denoise+JPEGXL"] = "sienna"
COLORSDICT["BPG"] = COLORSDICT["denoise+BPG"] = "wheat"


METHOD_TRANSLATIONS = {
    "lownoise": "JDC-Cn.9",
    "lownoise90": "JDC-Cn.9",
    "lownoise80": "JDC-Cn.8",
    "lownoise70": "JDC-Cn.7",
    "lownoise60": "JDC-Cn.6",
    "lownoise50": "JDC-Cn.5",
    "lownoise40": "JDC-Cn.4",
    "lownoise20": "JDC-Cn.2",
    "twomodels": "denoise+std AE",
    "lownoise_testolina": "JDC-Cn.8-Tdec",
    "std": "std AE",
    "testolina": "Testolina",
    "lownoise (90,ms-ssim)": "JDC-Cn.9-MS-SSIM",
    "lownoise (80,ms-ssim)": "JDC-Cn.8-MS-SSIM",
    "allnoise": "JDC-CN",
    "onlynoise": "JDC-N",
    "predenoise": "JDC-UD"
}


def parser_add_arguments(parser) -> None:
    """List and parser of arguments specific to this part of the project."""
    parser.add_argument(
        "--csv_fpath",
        help="Path to a single csv file to plot.",
    )
    parser.add_argument(
        "--csv_dpath",
        help="Path to a directory containing csv files to plot",
        default=RES_DPATH,
    )
    parser.add_argument(
        "--models_to_graph", nargs="*", help="Filter models to graph (if set)"
    )
    parser.add_argument(
        "--ignore_std", action="store_true", help="Std model is not part of x-y limit"
    )
    parser.add_argument(
        "--xrange",
        nargs="*",
        type=float,
        help="Range of x values to plot (none = autodetect based on learned models)",
    )
    parser.add_argument(
        "--yrange",
        nargs="*",
        type=float,
        help="Range of y values to plot (none = autodetect based on learned models)",
    )
    parser.add_argument(
        "--legend_ncol",
        type=int,
        default=1,
        help="Number of columns in legend (default: 1)",
    )
    parser.add_argument(
        "--legend_size",
        type=int,
        help="Legend font size",
    )

    parser.add_argument(
        "--graph_title",
        help="Graph title (optional)",
    )
    parser.add_argument(
        "--yscale",
        default="linear",
        help="Plot scale (linear, log, symlog, logit, ...)",
    )
    parser.add_argument('--fig_height', type=float, help='Output figure height', default=4.8)


def parser_autocomplete(args):
    """Autocomplete args after they've been parsed."""
    check_parameters(args)


def check_parameters(args):
    """Various assertions on parameters/args."""
    pass


def get_bpp(aresult: dict) -> float:
    for akey, aval in aresult.items():
        if akey.endswith("bpp"):
            return aval


def get_quality(aresult: dict) -> float:
    for akey, aval in aresult.items():
        if akey.endswith("psnr") or akey.lower().endswith("ssim"):
            return aval


def get_ylabel(results: list) -> str:
    for akey in results[0]:
        if akey.endswith("psnr") or akey.lower().endswith("ssim"):
            return akey


def is_learned(result: dict) -> bool:
    """Return True if result comes from a trained model."""
    try:
      if (
          "JPEG" in result["arch"]
          or "BPG" in result["arch"]
          or "JPEGXL" in result["arch"]
          # or "Passthrough" in result["arch"]
      ):
          return False
    except KeyError as e:
      print(f'graph_res.is_learned error: {e}')
      breakpoint()
    print(result)
    return True


def get_range(
    results: list,
    getter: Callable[[list], Union[str, float]],
    ignore_std: bool,
    argrange: Optional[tuple[int, int]] = None,
) -> tuple:
    """Get range of values to graph (+/- 10% of learned models)."""
    if argrange:
        return argrange
    range = [sys.maxsize, 0]
    for aresult in results:
        v = getter(aresult)
        if (
            (
                args.models_to_graph
                and aresult["train_method"] not in args.models_to_graph
            )
            or not is_learned(aresult)
            or "ssim" in aresult["train_method"].lower()
            or ("std" in aresult["train_method"] and ignore_std)
            or v == 0
            or np.isinf(v)
        ):
            # print(aresult)
            # breakpoint()
            continue
        # if v > 0.8:
        # breakpoint()
        if v > range[1]:
            range[1] = v
        if v < range[0]:
            range[0] = v
    # breakpoint()
    padding = abs(range[1] - range[0]) * 0.05
    range[0] = range[0] - padding
    range[1] = range[1] + padding
    return range


def get_bpp_range(
    results: list, ignore_std: bool, argrange: Optional[tuple[int, int]] = None
) -> tuple:
    """Get range of bitrate to graph (+/- 10% of learned models)."""
    return get_range(results, get_bpp, ignore_std, argrange)


def get_y_range(
    results: list, ignore_std: bool, argrange: Optional[tuple[int, int]] = None
) -> tuple:
    """Get range of quality to graph (+/- 10% of learned models)."""
    return get_range(results, get_quality, ignore_std, argrange)


def get_closest_xindex(target_bpp, bpps):
    dif = sys.maxsize
    best_i = None
    # breakpoint()
    for i, bpp in enumerate(bpps):
        if abs(bpp - target_bpp) < dif:
            dif = abs(bpp - target_bpp)
            best_i = i
    return i


def interpolate(xvals: list, yvals: list, range: tuple, method=None) -> tuple:
    bpps = np.linspace(range)
    losses = [None] * len(bpps)
    for i in len(xvals):
        losses[get_closest_xindex(xvals[i], bpps)] = yvals[i]
    return bpps, losses


def get_x_y_labels(results: list) -> tuple:
    x_y_vals = dict()
    for aresult in results:
        if 'arch' not in aresult:
            aresult['arch'] = args.arch  # not great hack, args shouldn't be accessed here
        # rename if necessary
        if not aresult["train_method"]:
            aresult["train_method"] = aresult["arch"]
        elif aresult["train_method"] == "twomodels" and aresult["arch"] in (
            "JPEG",
            "BPG",
            "JPEGXL",
            "Passthrough",
        ):
            aresult["train_method"] = (
                "denoise+" + aresult["arch"]
            )  # + "_" + aresult["train_method"]
        else:
            aresult["train_method"] = METHOD_TRANSLATIONS.get(
                aresult["train_method"], aresult["train_method"]
            )
        if aresult.get("lossf") == "msssim":  # in aresult["model_name"]:
            # breakpoint()
            if "SSIM" not in aresult["train_method"]:
                aresult["train_method"] = aresult["train_method"].replace(
                    ")", ", MS-SSIM)"
                )
            if "MS-SSIM" not in aresult["train_method"]:
                aresult["train_method"] += " (MS-SSIM)"
        # get values
        cur_x_y_vals = x_y_vals.get(aresult["train_method"], ([], []))
        cur_x_y_vals[0].append(get_bpp(aresult))
        cur_x_y_vals[1].append(get_quality(aresult))
        x_y_vals[aresult["train_method"]] = cur_x_y_vals
    cleanup_results(x_y_vals)
    for label, x_y in x_y_vals.items():
        yield x_y[0], x_y[1], label


def cleanup_results(x_y_vals: dict[str, tuple[list[float], list[float]]]) -> bool:
    """
    Remove oscillating results; keep best ones.

    Applies to JPEGXL
    """
    skipped = 0
    for label in x_y_vals:
        new_x_y_vals = ([], [])
        for x_candidate, y_candidate in zip(x_y_vals[label][0], x_y_vals[label][1]):
            for x_unfiltered, y_unfiltered in zip(
                x_y_vals[label][0], x_y_vals[label][1]
            ):
                if (
                    x_candidate > x_unfiltered
                    and y_candidate < y_unfiltered
                    and skipped < 3
                    and "JPEGXL" in label
                    and x_candidate < 0.2
                    # and "Testolina" not in label
                    # and "std" not in label
                ):
                    skipped += 1
                    break
            else:
                skipped = 0
                new_x_y_vals[0].append(x_candidate)
                new_x_y_vals[1].append(y_candidate)
        x_y_vals[label] = new_x_y_vals


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def open_csv(fpath: str) -> list:
    with open(fpath, "r") as fp:
        results = list(csv.DictReader(fp))
    for ares in results:
        for akey, aval in ares.items():
            if isfloat(aval):
                ares[akey] = float(aval)
    return results


def is_shown(xvals, yvals, xrange, yrange):
    # breakpoint()
    if xvals is not None and (min(xvals) > xrange[-1] or max(xvals) < xrange[0]):
        return False
    if min(yvals) > yrange[-1] or max(yvals) < yrange[0]:
        return False
    return True


if __name__ == "__main__":
    args = initfun.get_args(
        [parser_add_arguments],
        parser_autocomplete,
        def_config_fpaths=dc_common_args.DC_DEFCONF_FPATHS,
    )
    if args.csv_fpath:
        csv_fpaths = [args.csv_fpath]
    elif args.csv_dpath:
        csv_fpaths = (
            os.path.join(args.csv_dpath, fn) for fn in os.listdir(args.csv_dpath)
        )
    else:
        raise ValueError(
            "graph_res.py error: csv_fpath or csv_fpaths argument must be provided"
        )
    for csv_fpath in csv_fpaths:
        print(csv_fpath)
        if not csv_fpath.endswith(".csv") or "~lock" in csv_fpath:
            continue
        res = open_csv(csv_fpath)

        print(res)
        res.sort(key=get_bpp)

        # breakpoint()
        if args.fig_height:
            plt.figure(figsize=(6.4, args.fig_height))
        for xvals, yvals, label in get_x_y_labels(res):
            # breakpoint()
            range = get_bpp_range(res, ignore_std=args.ignore_std, argrange=args.xrange)
            if "Passthrough" in label:
                if is_shown(
                    None,
                    yvals,
                    range,
                    get_y_range(res, ignore_std=args.ignore_std, argrange=args.yrange),
                ):
                    plt.hlines(
                        yvals[0],
                        linestyles="dashed",
                        label="Noisy" if label == "Passthrough" else "Denoised",
                        xmin=range[0],
                        xmax=range[1],
                        colors="gray" if label == "Passthrough" else "silver",
                    )
                    # breakpoint()
                    continue
                # plt.hlines()
            if args.models_to_graph and label not in args.models_to_graph:
                continue
            # if label == "JPEGXL":
            #     breakpoint()
            plt.plot(
                xvals,
                yvals,
                marker=MARKERSDICT.get(label, "."),
                label=(
                    label
                    if is_shown(
                        xvals,
                        yvals,
                        range,
                        get_y_range(
                            res, ignore_std=args.ignore_std, argrange=args.yrange
                        ),
                    )
                    else "__nolegend__"
                ),
                linestyle="-." if label.startswith("denoise+") else "-",
                color=COLORSDICT.get(label, "black"),
            )
            # scaled_xvals, scaled_yvals = interpolate(xvals, yvals, range)
            # plt.plot(scaled_xvals, scaled_yvals, label=label)
        range = get_bpp_range(
            res, ignore_std=args.ignore_std, argrange=args.xrange
        )  # recompute after get_x_y_labels adjustments
        # breakpoint()
        plt.xlim(range)
        print(get_y_range(res, ignore_std=args.ignore_std, argrange=args.yrange))
        plt.ylim(get_y_range(res, ignore_std=args.ignore_std, argrange=args.yrange))
        # plt.ylim(0.90, 0.97)
        plt.xlabel("← bits per pixel (bpp)")
        # plt.ylabel(
        #     get_ylabel(res)
        #     .replace("_", " ")
        #     .replace("msssim", "MS-SSIM")
        #     .replace("lownoise", "low noise")
        #     .replace("q>=1.", "q=1.")
        # )
        if 'msssim' in get_ylabel(res):
            plt_quality_label = 'MS-SSIM'
        elif 'psnr' in get_ylabel(res):
            plt_quality_label = 'PSNR'
        else:
            raise NotImplementedError(f'Unknown quality label in {get_ylabel(res)}')
        plt.ylabel(f'{plt_quality_label} →')
        plt.grid(axis="both", which="both")
        plt.legend(
            fontsize="small",
            loc="lower right",
            ncol=args.legend_ncol,
            prop={"size": args.legend_size} if args.legend_size else None,
        )

        if args.graph_title:
            print(f'debug: ignoring title {args.graph_title}')
            # plt.title(args.graph_title)
        plt.yscale(args.yscale)
        if args.config:
            res_dpath = os.path.join(args.csv_dpath, utilities.get_leaf(args.config))
            os.makedirs(res_dpath, exist_ok=True)
        else:
            res_dpath = utilities.get_root(csv_fpath)
        plt.tight_layout()
        plt.savefig(os.path.join(res_dpath, utilities.get_leaf(csv_fpath) + ".pdf"))

        plt.close()

        plt.show()
        # breakpoint()
