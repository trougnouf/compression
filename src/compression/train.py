# -*- coding: utf-8 -*-
"""
Train a compression model.

See ../../config/compression/default.yaml for default parameters,
libs/initfun.py:parser_add_argument for parameters definition.
"""

import collections
import logging
import time
import os
import unittest
import sys
from typing import Optional, List
import torch

try:
    import torch.utils.tensorboard as tensorboardX
except ModuleNotFoundError:
    import tensorboardX
sys.path.append("..")
from common.extlibs import radam
from common.extlibs import pt_rangelars
from compression.libs import initfun  # argparse is here
from compression.libs import Meter
from common.libs import locking
from compression.libs import model_ops
from compression import tests
from compression.tools import cleanup_checkpoints
from common.libs import pt_helpers
from compression.libs import datasets

OPTIMIZERS = {
    "RangeLars": pt_rangelars.RangerLars,
    "RAdam": radam.RAdam,
    "Adam": torch.optim.Adam,
}
logger = logging.getLogger("ImageCompression")


def parser_add_arguments(parser) -> None:
    # useful config
    parser.add_argument("--tot_steps", type=int, help="Number of training steps")
    parser.add_argument("--reset_lr", action="store_true")
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument("--reset_global_step", action="store_true")
    # moderately useful config
    parser.add_argument("--base_lr", type=float)
    parser.add_argument(
        "--train_data_dpaths", nargs="*", type=str, help="Training image directories"
    )
    # very unusual config
    parser.add_argument(
        "--tot_epoch", type=int, help="Number of passes through the dataset"
    )
    parser.add_argument("--lr_update_mode", help="use worse_than_previous")
    parser.add_argument(
        "--lr_decay",
        type=float,
        help="LR is multiplied by this value whenever performance does not improve in an steps-epoch",
    )
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--print_freq", type=int, help="Logger frequency in # steps")
    parser.add_argument("--save_model_freq", type=int)
    parser.add_argument("--test_step", type=int)
    parser.add_argument(
        "--optimizer_init", help="Initial optimizer: " + str(OPTIMIZERS.keys())
    )
    parser.add_argument(
        "--optimizer_final", help="Final optimizer: " + str(OPTIMIZERS.keys())
    )
    parser.add_argument(
        "--optimizer_switch_step",
        type=int,
        help="# of steps after which optimizer_final is chosen",
    )

    parser.add_argument(
        "--lr_switches",
        type=float,
        nargs="*",
        help="Ratio of steps after which LR decays. Normally set to None and LR decays with stagnation.",
    )
    parser.add_argument(
        "--add_artificial_noise",
        action="store_true",
        help="Train denoising with artificial noise. Poor performance compared to DC; for comparison purpose only.",
    )


def parser_autocomplete(args):
    args.pretrain_prefix = "val" if args.val_dpath is not None else "test"
    check_parameters(args)


def check_parameters(args):
    assert args.optimizer_final is None or args.optimizer_final in OPTIMIZERS
    assert (
        args.test_std_dpaths is not None and args.val_dpath is not None
    ), "test_std_dpaths and val_dpath are required"


def train(
    model,
    train_loader,
    test_loaders,
    val_loader,
    device,
    tb_logger,
    data_epoch,
    global_step,
    jsonsaver,
    optimizer,
    config,
):
    """
    Train model for a data epoch
    returns current step
    """
    logger.info("Data epoch {} begin".format(data_epoch))
    model.train()
    elapsed, losses, bpps, bpp_features, bpp_zs, visual_losses = [
        Meter.AverageMeter(config["print_freq"]) for _ in range(6)
    ]
    used_dists_all = set()

    vis_loss_hist = collections.deque(maxlen=2)
    bpp_loss_hist = collections.deque(maxlen=2)
    loss_hist = collections.deque(maxlen=2)
    assert val_loader and test_loaders
    # test_prefix = "test" if val_loader is None else "val"
    # val_test_loader = test_loader if val_loader is None else val_loader
    for batch_idx, input in enumerate(train_loader):
        input = input.to(device)
        locking.check_pause()
        start_time = time.time()
        global_step += 1
        # print("debug", torch.max(input), torch.min(input))
        # (
        #     clipped_recon_image,
        #     visual_loss,
        #     bpp_feature,
        #     bpp_z,
        #     bpp,
        #     used_dists_now,
        #     flag,
        # ) = model(input)

        model_out = model(input)

        distribution_loss = model_out["bpp"]
        if config["num_distributions"] <= 16 and "used_dists" in model_out:
            used_dists_all.update(model_out["used_dists"])

        distortion = model_out["visual_loss"]
        rd_loss = config["train_lambda"] * distortion + distribution_loss
        optimizer.zero_grad()
        rd_loss.backward()

        if hasattr(model, "clip_gradient"):
            clip_gradient = model.clip_gradient
        else:

            def clip_gradient(optimizer, grad_clip):
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.data.clamp_(-grad_clip, grad_clip)

        clip_gradient(optimizer, 5)
        optimizer.step()
        # model_time += (time.time()-start_time)

        if (global_step % config["print_freq"]) == 0:
            # These were a separate step (global_step % cal_step), but we are
            # no longer calculating the average so this step should be simplified
            elapsed.update(time.time() - start_time)
            losses.update(rd_loss.item())
            bpps.update(model_out["bpp"].item())
            if "bpp_feature" in model_out:
                bpp_features.update(model_out["bpp_feature"].item())
                tb_logger.add_scalar("bpp_feature", bpp_features.avg, global_step)
            if "bpp_sidestring" in model_out:
                bpp_zs.update(model_out["bpp_sidestring"].item())
                tb_logger.add_scalar("bpp_z", bpp_zs.avg, global_step)
            visual_losses.update(model_out["visual_loss"].item())
            # begin = time.time()
            tb_logger.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
            tb_logger.add_scalar("rd_loss", losses.avg, global_step)
            tb_logger.add_scalar("visual_loss", visual_losses.avg, global_step)
            # tb_logger.add_scalar('psnr', psnrs.avg, global_step)
            tb_logger.add_scalar("bpp", bpps.avg, global_step)

            process = global_step / config["tot_steps"] * 100.0
            log = " | ".join(
                [
                    f'{config["expname"]}',
                    f'Step [{global_step}/{config["tot_steps"]}={process:.2f}%]',
                    f"Data Epoch {data_epoch}",
                    f"Time {elapsed.val:.3f} ({elapsed.avg:.3f})",
                    f'Lr {optimizer.param_groups[0]["lr"]}',
                    f"Total Loss {losses.val:.3f} ({losses.avg:.3f})",
                    # f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f"Bpp {bpps.val:.5f} ({bpps.avg:.5f})",
                    # f"Bpp_feature {bpp_features.val:.5f} ({bpp_features.avg:.5f})",
                    # f"Bpp_z {bpp_zs.val:.5f} ({bpp_zs.avg:.5f})",
                    f"Visual loss {visual_losses.val:.5f} ({visual_losses.avg:.5f})",
                ]
            )
            if "used_dists" in model_out:
                if config["num_distributions"] <= 16:
                    log += "| used_dists: {}".format(used_dists_all)
                    used_dists_all = set()
                else:
                    log += "| used_dists: {}".format(len(model_out["used_dists"]))
                    tb_logger.add_scalar(
                        "num_used_dists", len(model_out["used_dists"]), global_step
                    )
                if "flag" in model_out:
                    log += "| flag: {}".format(model_out["flag"])

            logger.info(log)
        if (global_step % config["save_model_freq"]) == 0:
            jsonsaver.add_res(
                global_step,
                {
                    "train_bpp": bpps.avg,
                    "train_visual_loss": visual_losses.avg,
                    "train_bpp_string": bpp_features.avg,
                    "train_bpp_side_string": bpp_zs.avg,
                    "train_combined_loss": losses.avg,
                    "lr_vis": optimizer.param_groups[0]["lr"],
                    "lr_bpp": optimizer.param_groups[-1]["lr"],
                },
            )
            model_ops.save_model(
                model,
                global_step,
                os.path.join(config["save_path"], "saved_models"),
                optimizer=optimizer,
            )

        if (global_step % config["val_step"]) == 0 or (
            global_step % config["test_step"]
        ) == 0:
            # VAL
            if (global_step % config["val_step"]) == 0:
                val_vis_loss, val_bpp_loss, val_combined_loss = tests.test_dir(
                    model=model,
                    step=global_step,
                    jsonsaver=jsonsaver,
                    config=config,
                    device=device,
                    prefix="val",
                    loader=val_loader,
                    tb_logger=tb_logger,
                    crop_to_multiple_size=args.test_crop_mult,
                    incl_combined_loss=True,
                )

                update_vis = update_bpp = False
                if len(loss_hist) > 0 and max(loss_hist) < val_combined_loss:
                    if max(vis_loss_hist) < val_vis_loss:
                        update_vis = True
                    if max(bpp_loss_hist) < val_bpp_loss:
                        update_bpp = True
                    model_ops.adjust_learning_rate(
                        optimizer,
                        training_progress=global_step / config["tot_steps"],
                        lr_decay=config["lr_decay"],
                        init_lr=config["base_lr"],
                        bit=update_bpp,
                        encdec=update_vis,
                        all=bool(config["lr_switches"]),
                        lr_switches=config["lr_switches"],
                    )
                vis_loss_hist.append(val_vis_loss)
                bpp_loss_hist.append(val_bpp_loss)
            # /VAL
            # TEST
            if (global_step % config["train_step"]) == 0:
                for test_name, test_loader in test_loaders.items():
                    tests.test_dir(
                        model=model,
                        step=global_step,
                        jsonsaver=jsonsaver,
                        config=config,
                        device=device,
                        prefix=f"test_{test_name}",
                        loader=test_loader,
                        tb_logger=tb_logger,
                        crop_to_multiple_size=args.test_crop_mult,
                        incl_combined_loss=True,
                    )
            model.train()
        if (global_step % config["save_model_freq"]) == 0:
            cleanup_checkpoints.cleanup_checkpoints(expname=config["expname"])

    jsonsaver.add_res(
        global_step,
        {
            "train_bpp": bpps.avg,
            "train_visual_loss": visual_losses.avg,
            "train_bpp_string": bpp_features.avg,
            "train_bpp_side_string": bpp_zs.avg,
            "train_combined_loss": losses.avg,
        },
    )
    model_ops.save_model(
        model,
        global_step,
        os.path.join(config["save_path"], "saved_models"),
        optimizer=optimizer,
    )
    tests.test_dir(
        model=model,
        step=global_step,
        jsonsaver=jsonsaver,
        config=config,
        loader=val_loader,
        prefix="val",
        device=device,
        tb_logger=tb_logger,
        crop_to_multiple_size=args.test_crop_mult,
    )
    model.train()

    return global_step


def test():
    pass


def val():
    pass


# TODO check that pretrain is checked against None not ''

# TODO check args


def train_handler(args, jsonsaver, device):
    global_step, model = model_ops.get_step_and_loaded_model(vars(args), device)

    # get test loader(s)
    val_loader, test_loaders = datasets.get_val_test_loaders(
        args.val_dpath, args.test_std_dpaths
    )

    # get optimizer
    if hasattr(model, "get_optimizer"):
        optimizer = model.get_optimizer()
        assert hasattr(
            model, "load_weights"
        )  #  the model will also load its own optimizer weights
    else:
        optimizer = OPTIMIZERS[args.optimizer_init]
        is_init_optimizer = True
        if (
            args.optimizer_final is not None
            and args.optimizer_switch_step <= global_step
        ):
            optimizer = OPTIMIZERS[args.optimizer_final]
            print("optimizer: {}".format(str(optimizer)))
            is_init_optimizer = False
        optimizer = optimizer(model.get_parameters(lr=args.base_lr), lr=args.base_lr)
        # optimizer = optim.Adam(parameters, lr=base_lr)
        if args.pretrain is not None and not args.reset_optimizer:
            logger.info("loading optimizer:{}".format(args.pretrain + ".opt"))
            model_ops.load_model(optimizer, args.pretrain + ".opt", device=device)
            # if os.path.isfile(args.pretrain+'.opt.module'):
            #     optimizer = torch.load(args.pretrain+'.opt.module', map_location=device)
            if args.reset_lr:
                model_ops.reset_lr(optimizer, model, args.base_lr)
    # FIXME freeze_autoencoder is likely wrong; should replace the optimizer with one that doesn't have the AE parameters
    if args.freeze_autoencoder or (
        args.freeze_autoencoder_steps is not None
        and args.freeze_autoencoder_steps > global_step
    ):
        logger.info("Freezing autoencoder (experimental)")
        model.freeze_autoencoder()
    # global train_loader
    tb_logger = tensorboardX.SummaryWriter(os.path.join(args.save_path, "events"))
    train_dataset = datasets.Datasets(
        args.train_data_dpaths,
        args.image_size,
        add_artificial_noise=args.add_artificial_noise,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        num_workers=args.batch_size,
        drop_last=True,
    )
    steps_epoch = global_step // (len(train_dataset) // (args.batch_size))
    print(f"save_path: {args.save_path}")
    for data_epoch in range(steps_epoch, args.tot_epoch):
        # set stage
        model_ops.set_model_stage(
            model, global_step, args.stage_switches, args.tot_steps
        )

        # get optimizer
        if hasattr(model, "get_optimizer"):
            optimizer = model.get_optimizer()
        elif (
            args.optimizer_final is not None
            and args.optimizer_switch_step <= global_step
        ):
            if is_init_optimizer:
                logger.info(
                    "Switching optimizer from {} to {}".format(
                        args.optimizer_init, args.optimizer_final
                    )
                )
                optimizer = OPTIMIZERS[args.optimizer_final](
                    model.get_parameters(lr=args.base_lr), lr=args.base_lr
                )
                is_init_optimizer = False
        if global_step > args.tot_steps:
            logger.info("Ending at global_step={}".format(global_step))
            break
        if (
            args.freeze_autoencoder_steps is not None
            and model.frozen_autoencoder
            and global_step >= args.freeze_autoencoder_steps
        ):
            model.unfreeze_autoencoder()
            logger.info("unFreezing autoencoder")
            # def train(model, train_loader, test_loader, val_loader, device, tb_logger, data_epoch, global_step, jsonsaver, optimizer, config):

        global_step = train(
            model=model,
            data_epoch=data_epoch,
            global_step=global_step,
            jsonsaver=jsonsaver,
            optimizer=optimizer,
            config=vars(args),
            train_loader=train_loader,
            test_loaders=test_loaders,
            val_loader=val_loader,
            device=device,
            tb_logger=tb_logger,
        )
        cleanup_checkpoints.cleanup_checkpoints(expname=args.expname)
        # save_model(model, global_step, save_path)


class Test_train(unittest.TestCase):
    """
    [compression]$ python -m unittest discover .
    or
    [compression]$ python -m unittest train.py
    TODO use a subset of the dataset s.t. it doesn't take an unrealistic amount of time to run
    unittest yields "ResourceWarning: unclosed file <_io.BufferedReader" which doesn't occur in normal runs.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_train_from_scratch_and_load(self):
        # train from scratch
        args = initfun.get_args(
            parser_add_arguments,
            parser_autocomplete,
            [
                "--num_distributions",
                "64",
                "--arch",
                "Balle2017ManyPriors",
                "--tot_steps",
                "5000",
                "--train_lambda",
                "128",
                "--expname",
                "unittest_scratch",
            ],
        )
        jsonsaver = initfun.get_jsonsaver(args)
        device = pt_helpers.get_device(args.device)
        train_handler(args, jsonsaver, device)
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    "..",
                    "..",
                    "models",
                    "compression",
                    "unittest_scratch",
                    "trainres.json",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    "..",
                    "..",
                    "models",
                    "compression",
                    "unittest_scratch",
                    "saved_models",
                    "iter_5000.pth",
                )
            )
        )
        # load that model
        # train from scratch
        args = initfun.get_args(
            parser_add_arguments,
            parser_autocomplete,
            [
                "--num_distributions",
                "64",
                "--arch",
                "Balle2017ManyPriors",
                "--tot_steps",
                "5000",
                "--train_lambda",
                "64",
                "--pretrain",
                "unittest_scratch",
                "--expname",
                "unittest_load",
                "--reset_global_step",
            ],
        )
        jsonsaver = initfun.get_jsonsaver(args)
        device = pt_helpers.get_device(args.device)
        train_handler(args, jsonsaver, device)
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    "..",
                    "..",
                    "models",
                    "compression",
                    "unittest_load",
                    "trainres.json",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    "..",
                    "..",
                    "models",
                    "compression",
                    "unittest_load",
                    "saved_models",
                    "iter_5000.pth",
                )
            )
        )


if __name__ == "__main__":
    args = initfun.get_args(parser_add_arguments, parser_autocomplete)
    jsonsaver = initfun.get_jsonsaver(args)
    device = pt_helpers.get_device(args.device)
    train_handler(args, jsonsaver, device)
