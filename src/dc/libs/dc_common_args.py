"""Handle arguments common to different parts of the DC project."""
import os

MODELS_DPATH = os.path.join("..", "..", "models", "dc")
DC_DEFCONF_FPATHS = [
    os.path.join("..", "dc", "config", "defaults.yaml"),
]


def parser_add_arguments(parser) -> None:
    """Add arguments common to the DC project to the parser."""
    parser.add_argument(
        "--denoise_test_reserve", nargs="*", help="noise/gt pairs not used for training"
    )
    parser.add_argument("--test_denoise_dpath")
    parser.add_argument(
        "--quality_whole_csv_fpath",
        type=str,
        help="Path to .csv file containing image paths scores.",
    )
    parser.add_argument(
        "--min_quality",
        type=float,
        help=(
            "Min MS-SSIM score of training images. If set, uses PickyDenoisingDatasetFromList "
            "(used with quality_crops_csv_fpaths)"
        ),
    )
    parser.add_argument("--test_on_cpu", action="store_true")
