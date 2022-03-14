"""Model initialization functions."""

from typing import Optional, List
import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter

sys.path.append("..")
from common.extlibs import pt_ms_ssim
from common.libs import pt_helpers
from compression.libs import initfun
from compression.models import *
from compression.models import abstract_model
from compression.models.coarse2fine_compressor import Coarse2Fine_ImageCompressor
from compression.models.Balle2018PT_compressor import *
from compression.models.standard_compressor import *
from compression.models.testolina_compressor import *
from compression.models.passthrough import Passthrough_ImageCompressor

try:
    from compression.models.PixelShuffle4c_compressor import *
except ModuleNotFoundError:
    pass
try:
    from compression.models.ESPCN_compressor import *
except ModuleNotFoundError:
    pass
try:
    from compression.models.hardnet_compressor import *
except ModuleNotFoundError:
    pass
from compression.models.manynets_compressor import *

try:
    from compression.models.SIREN_compressor import *
except ModuleNotFoundError:
    pass
try:
    from compression.models.quadtree_compressor import *
except ModuleNotFoundError:
    pass
try:
    from dc.models.manynets_dc import *
except ModuleNotFoundError:
    pass
try:
    from dc.models.Balle20182xL_compressor import *
except ModuleNotFoundError:
    pass
# ARCHS = 'Balle2018PT', 'PixelShuffle4c', 'ESPCNsynth'  # TODO gen from imports_ImageCompressor
# TODO arch search: disable bitrate estimation / loss

logger = logging.getLogger("ImageCompression")


def save_model(model, iter, save_dpath, optimizer=None):
    """Save model and optimizer to disk.

    filename is iter_ITERATION.pth.SUFFIX
    """
    if hasattr(model, "get_weights"):
        for weights, suffix in model.get_weights():
            out_fpath = os.path.join(save_dpath, f"iter_{iter}.pth")
            if suffix is not None:
                out_fpath += f".{suffix}"
            torch.save(weights.state_dict(), out_fpath)
    else:
        fn = "iter_{}.pth".format(iter)
        torch.save(model.state_dict(), os.path.join(save_dpath, fn))
        # torch.save(model, os.path.join(save_dpath, fn + ".module"))
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(save_dpath, fn + ".opt"))
            # torch.save(optimizer, os.path.join(save_dpath, fn + ".opt.module"))


def load_model(model, fpath, device="cuda:0"):
    """
    Load a model's weights from a given fpath dictionary.

    step is infered from fpath
    """
    if hasattr(model, "load_weights"):
        # if the model has its own weights loading method, let it do.
        model.load_weights(fpath)
    else:
        device = pt_helpers.get_device(device)
        try:
            with open(fpath, "rb") as f:
                pretrained_dict = torch.load(f, map_location=device)
                model_dict = model.state_dict()
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict
                }
                model_dict.update(pretrained_dict)
                print(
                    model.load_state_dict(model_dict)
                )  # , strict=False))  # strict doesn't work for optimizer
        except RuntimeError as e:
            print(e)
            with open(fpath, "rb") as f:
                pretrained_dict = torch.load(f, map_location=device)
                model_dict = model.state_dict()
                if "bitEstimators" in str(e):
                    pretrained_dict = {
                        k: v
                        for k, v in pretrained_dict.items()
                        if (k in model_dict and "bitEstimators" not in k)
                    }
                else:
                    breakpoint()
                model_dict.update(pretrained_dict)
                print(
                    model.load_state_dict(model_dict)
                )  # , strict=False))  # strict doesn't work for optimizer
    # f = str(f)
    if fpath.find("iter_") != -1 and fpath.find(".pth") != -1:
        st = fpath.find("iter_") + 5
        ed = fpath.find(".pth", st)
        return int(fpath[st:ed])
    else:
        return 0


def reset_lr(optimizer, model, base_lr):
    #  FIXME not done for Coarse2Fine-type
    model_parameters = model.get_parameters(lr=base_lr)
    for param_group in optimizer.param_groups:
        new_lr = None
        for model_parameter in model_parameters:
            if param_group["name"] == model_parameter["name"]:
                if "lr" in model_parameter:
                    new_lr = model_parameter["lr"]
                else:
                    new_lr = base_lr
                continue
        if new_lr is None:
            new_lr = base_lr
        logger.info(
            "reset_lr: reset lr of {} to {}".format(param_group["name"], new_lr)
        )
        param_group["lr"] = new_lr


class CustomModel_ImageCompressor(abstract_model.AbstractHyperpriorImageCompressor):
    def __init__(
        self,
        synthesis_prior_arch,
        synthesis_arch,
        analysis_arch,
        analysis_prior_arch,
        out_channel_N=192,
        out_channel_M=320,
        lossf="mse",
        device="cuda:0",
    ):
        super().__init__(
            out_channel_N=out_channel_N,
            out_channel_M=out_channel_M,
            lossf=lossf,
            device=device,
        )
        self.Encoder = globals()[analysis_arch](
            out_channel_N=out_channel_N, out_channel_M=out_channel_M
        )
        self.Decoder = globals()[synthesis_arch](
            out_channel_N=out_channel_N, out_channel_M=out_channel_M
        )
        self.priorDecoder = globals()[synthesis_prior_arch](
            out_channel_N=out_channel_N, out_channel_M=out_channel_M
        )
        self.priorEncoder = globals()[analysis_prior_arch](
            out_channel_N=out_channel_N, out_channel_M=out_channel_M
        )


def init_model(
    arch: str, device="cuda:0", std_suffix: str = "ImageCompressor", **kwargs
):
    """
    Initialize a compression model.

    std_suffix is 'ImageCompressor' (for most compression models) or 'DC' (for
    denoisecompress)
    """
    arch_class_str = std_suffix
    device = pt_helpers.get_device(device)
    if arch is not None and arch != "ImageCompressor" and arch != "DC":
        arch_class_str = "{}_{}".format(arch, arch_class_str)
    try:
        return globals()[arch_class_str](device=device, **kwargs).to(device)
    except KeyError as e:
        print(e)
        print("Invalid architecture: {}. List of valid architectures:".format(arch))
        for arch in globals():
            if "_" + std_suffix in arch:
                print(arch.split("_" + std_suffix)[0])
        exit(-1)


def set_model_stage(
    model, step: int, stage_switches: Optional[List[float]], tot_steps: int
):
    """Set a model's stage if needed, eg with Coarse2Fine models."""
    if hasattr(model, "stage") and stage_switches:
        res_stage = 1
        for cur_stage, switch_ratio in enumerate(stage_switches, start=1):
            if step >= switch_ratio * tot_steps:
                res_stage = cur_stage + 1
        if model.stage != res_stage:
            logger.info(
                f"model_ops.set_model_stage: switch from {model.stage} to {res_stage}"
            )
            model.stage = res_stage


def adjust_learning_rate(
    optimizer,
    training_progress: float,
    lr_decay: float,
    init_lr: float,
    bit=False,
    encdec=False,
    all=False,
    lr_switches: Optional[List[int]] = None,
):
    """Adjust the learning rate of part or all of a model."""
    for param_group in optimizer.param_groups:
        # check if we are updating a specific lr and this isn't it
        if not all:
            if ("bit" in param_group["name"] and not bit) or (
                "bit" not in param_group["name"] and not encdec
            ):
                continue
        if lr_switches:
            n_switches = sum(
                [training_progress >= threshold for threshold in lr_switches]
            )
            new_lr = init_lr * lr_decay ** n_switches
        else:
            new_lr = param_group["lr"] * lr_decay
        old_lr = param_group["lr"]
        if old_lr != new_lr:
            logger.info(
                "adjust_learning_rate: {param_group['name']}: {old_lr}->{new_lr}"
            )
        param_group["lr"] = new_lr


def get_step_and_loaded_model(config, device):
    """
    This method initializes a model , finds the best step, and return it along
    with the loaded model.
    """
    global_step = 0
    model = init_model(**config)
    if config["pretrain"] is not None and not os.path.isfile(config["pretrain"]):
        config["pretrain"] = initfun.get_best_checkpoint(
            exp=config["pretrain"],
            prefix=config["pretrain_prefix"],
            checkpoints_dir=config["checkpoints_dpath"],
            step=config.get("global_step"),
        )
        if config["pretrain"] == "":
            print(
                'error: config["pretrain"] is an empty string. Check that model and its trainres.json exist.'
            )
            breakpoint()
    if config["pretrain"] is not None:
        logger.info("loading model:{}".format(config["pretrain"]))
        if config.get("reset_global_step"):
            load_model(model, config["pretrain"], device=device)
        else:
            global_step = load_model(model, config["pretrain"], device=device)
    # set stage
    set_model_stage(
        model,
        global_step,
        config["stage_switches"],
        config.get("tot_steps", global_step),
    )
    return global_step, model


# # Keep this as-is for compatibility
# class ImageCompressor(nn.Module):
#     def __init__(self, out_channel_N=192, out_channel_M=320, lossf='mse', device='cuda:0'):
#         super(ImageCompressor, self).__init__()
#         self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
#         self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
#         self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
#         self.priorDecoder = Synthesis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
#         self.bitEstimator_z = BitEstimator(out_channel_N)
#         self.out_channel_N = out_channel_N
#         self.out_channel_M = out_channel_M
#         self.lossf = lossf
#         if lossf == 'mse':
#             self.lossfun = F.mse_loss
#         elif lossf == 'ssim':
#             self.lossclass = pt_ms_ssim.SSIM()
#             self.lossfun = self.lossclass.lossfun
#         elif lossf == 'msssim':
#             self.lossclass = pt_ms_ssim.MS_SSIM()
#             self.lossfun = self.lossclass.lossfun
#         else:
#             raise ValueError(lossf)
#         self.device = device

#         #self.visual_loss =

#     def forward(self, input_image):
#         quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 16, input_image.size(3) // 16, device=self.device)
#         quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 64, input_image.size(3) // 64, device=self.device)
#         quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
#         quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
#         feature = self.Encoder(input_image)
#         batch_size = feature.size()[0]

#         z = self.priorEncoder(feature)
#         if self.training:
#             compressed_z = z + quant_noise_z
#         else:
#             compressed_z = torch.round(z)
#         recon_sigma = self.priorDecoder(compressed_z)
#         feature_renorm = feature
#         if self.training:
#             compressed_feature_renorm = feature_renorm + quant_noise_feature
#         else:
#             compressed_feature_renorm = torch.round(feature_renorm)
#         recon_image = self.Decoder(compressed_feature_renorm)
#         # recon_image = prediction + recon_res
#         clipped_recon_image = recon_image.clamp(0., 1.)
#         # distortion
#         #visual_loss = torch.mean((recon_image - input_image).pow(2))
#         visual_loss = self.lossfun(recon_image, input_image)
#         # TODO add msssim

#         def feature_probs_based_sigma(feature, sigma):
#             mu = torch.zeros_like(sigma)
#             sigma = sigma.clamp(1e-10, 1e10)
#             gaussian = torch.distributions.laplace.Laplace(mu, sigma)
#             probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
#             total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
#             return total_bits, probs

#         def iclr18_estimate_bits_z(z):
#             prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
#             total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
#             return total_bits, prob

#         total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
#         total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
#         im_shape = input_image.size()
#         bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
#         bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
#         bpp = bpp_feature + bpp_z
#         return clipped_recon_image, visual_loss, bpp_feature, bpp_z, bpp

#     def compress(self, input_image):
#         feature = self.Encoder(input_image)
#         z = self.priorEncoder(feature)
#         compressed_z = torch.round(z)
#         compressed_feature = torch.round(feature)
#         return compressed_feature, compressed_z
