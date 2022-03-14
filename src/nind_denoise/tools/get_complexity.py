import ptflops
import sys

sys.path.append("..")
from nind_denoise.networks import UtNet


def complexity_analysis(model_class):
    """
    egrun:
        CUDA_AVAILABLE_DEVICES="" python train.py --test_flags complexity --pretrain mse_4096_b2017manypriors_64pr_16px_adam_2upd_d --device -1

    """
    IMGCH, IMGHEIGHT, IMGWIDTH = 3, 504, 504
    # TODO add custom GDN hook
    # res = encoder(test_img)
    # breakpoint()
    # print(test_img.shape)
    macs, params = ptflops.get_model_complexity_info(
        model_class,
        (IMGCH, IMGHEIGHT, IMGWIDTH),
    )


if __name__ == "__main__":
    model_class = UtNet.UtNet()
    complexity_analysis(model_class)


"""
results_:
31.032 M, 100.000% Params, 169.303 GMac, 100.000% MACs,
-> effective crop size is 480x480 ->
((3840×2160)÷(480×480))×169.303 = 6094.908
"""
