# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
# pylint: disable=no-member
# pylint: disable=redefined-outer-name
"""
Train a denoise+compression model.

See ../../config/dc/default.yaml for default parameters, libs/initfun.py:parser_add_argument for
parameters definition.

TODO merge with compression/train.py, separate testing from training.

sanity test:
    python train.py --val_step 500 --test_step 1500 --test_on_cpu
"""

import logging
import time
import os
import collections
import unittest
import sys
import torch

try:
    import torch.utils.tensorboard as tensorboardX
except ModuleNotFoundError:
    import tensorboardX
sys.path.append("..")
from common.libs import locking, pt_helpers
from common.extlibs import radam, pt_rangelars
from compression import tests
from compression.libs import initfun, Meter, model_ops, datasets
from dc.libs import dcdatasets
from compression.tools import cleanup_checkpoints
from nind_denoise import dataset_torch_3
from dc.libs import dc_common_args

DC_DEFTRAIN_FPATHS = [os.path.join("..", "dc", "config", "dctrain.yaml")]

VAL_NIMGS = 15

OPTIMIZERS = {
    "RangeLars": pt_rangelars.RangerLars,
    "RAdam": radam.RAdam,
    "Adam": torch.optim.Adam,
}
logger = logging.getLogger("ImageCompression")


def parser_add_arguments(parser) -> None:
    """List and parser of arguments specific to this part of the project"""
    # useful config
    parser.add_argument("--tot_steps", type=int, help="Number of training steps")
    parser.add_argument("--reset_lr", action="store_true")
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument("--reset_global_step", action="store_true")

    parser.add_argument(
        "--quality_crops_csv_fpaths",
        nargs="*",
        help=(
            "Path to .csv file(s) containing training image paths scores, used with min_quality."
            "(xpath, ypath, score)"
        ),
    )

    # moderately useful config
    parser.add_argument("--base_lr", type=float)
    # parser.add_argument(
    #     "--train_data_dpaths", nargs="*", type=str, help="Training image directories"
    # )  # TODO is this used anywhere?? or train_std_data_dpaths
    # very unusual config
    parser.add_argument(
        "--tot_epoch", type=int, help="Number of passes through the dataset"
    )
    parser.add_argument("--lr_update_mode", help="use worse_than_previous")
    parser.add_argument(
        "--lr_decay",
        type=float,
        help=(
            "LR is multiplied by this value whenever performance does not improve in a"
            "<patience> number of steps/epoch"
        ),
    )
    parser.add_argument("--batch_size_denoise", type=int)
    parser.add_argument("--batch_size_std", type=int)
    parser.add_argument("--print_freq", type=int, help="Logger frequency in # steps")
    parser.add_argument("--save_model_freq", type=int)
    parser.add_argument("--test_step", type=int)
    parser.add_argument("--val_step", type=int)
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
    # parser.add_argument('--denoising_ratio', type=float,
    #                     help='Ratio of noise/gt pairs to gt only')
    parser.add_argument(
        "--train_denoise_data_dpaths",
        nargs="*",
        help="noise/gt pairs path (eg: ../../datasets/resized/NIND_256_96)",
    )
    parser.add_argument("--train_std_data_dpaths", nargs="*")

    parser.add_argument(
        "--val_std_dpath",
        help=(
            "Validation data path for standard training."
            " Not used when the model is trained to denoise+compress."
        ),
    )
    parser.add_argument("--ds_opts", nargs="*", help="Dataset options (eg: nind_noisy)")
    parser.add_argument(
        "--lr_switches",
        type=float,
        nargs="*",
        help="Ratio of steps after which LR decays. Normally set to None and LR decays with stagnation.",
    )
    parser.add_argument(
        "--add_artificial_noise",
        action="store_true",
        help="Add artificial noise as in Testolina et al. 2021",
    )


def parser_autocomplete(args):
    """Autocomplete args after they've been parsed."""
    args.denoise_training = args.batch_size_denoise > 0
    args.std_training = args.batch_size_std > 0
    if args.ds_opts is None:
        args.ds_opts = []
    check_parameters(args)


def check_parameters(args):
    """Various assertions on parameters/args."""
    assert args.optimizer_final is None or args.optimizer_final in OPTIMIZERS
    #  assert args.test_dpath is not None or args.val_dpath is not None, \
    #  'test_dpath and/or val_dpath required to update lr'
    # if args.denoise_training is False:
    #    assert args.val_step == args.kodak_test_step


def train(
    model,
    train_std_loader,
    train_denoise_loader,
    test_std_loaders,
    val_loader,
    test_denoise_loader,
    device,
    tb_logger,
    data_epoch,
    global_step,
    jsonsaver,
    optimizer,
    config,
    loss_cls,
):
    """
    Train model for a data epoch.

    returns current step
    """
    logger.info("Data epoch %s begin", data_epoch)
    model.train()
    (
        elapsed,
        losses,
        bpps,
        bpp_features,
        bpp_zs,
        visual_losses,
        reconstruction_losses,
    ) = [Meter.AverageMeter(config["print_freq"]) for _ in range(7)]
    used_dists_all = set()
    # assert (not hasattr(model, "lossfun")) or model.lossfun is None

    vis_loss_hist = collections.deque(maxlen=2)
    bpp_loss_hist = collections.deque(maxlen=2)
    loss_hist = collections.deque(maxlen=2)
    #  test_prefix = 'test' if val_loader is None else 'val'
    #  Data loaders
    if config["denoise_training"] and config["std_training"]:
        train_loaders = zip(train_std_loader, train_denoise_loader)
    elif config["denoise_training"]:
        train_loaders = train_denoise_loader
    elif config["std_training"]:
        train_loaders = train_std_loader
    for _, in_batch in enumerate(train_loaders):
        locking.check_pause()
        # in_batch = in_batch.to(device)
        # TODO merge in_batchs&
        if config["denoise_training"] and config["std_training"]:
            (std_batch_x, std_batch_y), (nind_batch_x, nind_batch_y) = in_batch
            gt_batch = torch.cat((std_batch_x, nind_batch_x), dim=0)
            in_batch = torch.cat((std_batch_y, nind_batch_y), dim=0)
        elif config["denoise_training"]:
            gt_batch, in_batch = in_batch
        elif config["std_training"]:
            gt_batch = in_batch = in_batch
        in_batch = in_batch.to(device)
        gt_batch = gt_batch.to(device)

        start_time = time.time()
        global_step += 1
        # print("debug", torch.max(in_batch), torch.min(in_batch))

        # clipped_recon_image, _, bpp_feature, bpp_z, bpp, used_dists_now, flag = model(
        #    in_batch
        # )
        model_out = model(in_batch)
        distribution_loss = model_out["bpp"]
        if config["num_distributions"] <= 16 and "used_dists" in model_out:
            used_dists_all.update(model_out["used_dists"])

        distortion = loss_cls(gt_batch, model_out["reconstructed_image"])
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
            reconstruction_loss = loss_cls(in_batch, model_out["reconstructed_image"])
            reconstruction_losses.update(reconstruction_loss.item())
            losses.update(rd_loss.item())
            bpps.update(model_out["bpp"].item())
            if "bpp_feature" in model_out:
                bpp_features.update(model_out["bpp_feature"].item())
                tb_logger.add_scalar("bpp_feature", bpp_features.avg, global_step)
            if "bpp_sidestring" in model_out:
                bpp_zs.update(model_out["bpp_sidestring"].item())
                tb_logger.add_scalar("bpp_z", bpp_zs.avg, global_step)
            visual_losses.update(distortion.item())

            # begin = time.time()
            # TODO deprecate tensorboard
            tb_logger.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
            tb_logger.add_scalar("rd_loss", losses.avg, global_step)
            tb_logger.add_scalar("visual_loss", visual_losses.avg, global_step)
            # tb_logger.add_scalar('psnr', psnrs.avg, global_step)
            tb_logger.add_scalar("bpp", bpps.avg, global_step)
            progress = global_step / config["tot_steps"] * 100.0
            log = " | ".join(
                [
                    f'{config["expname"]}',
                    f'Step [{global_step}/{config["tot_steps"]}={progress:.2f}%]',
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
            if "flag" in model_out:  # safe to rm?
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
                    "reconstruction_loss": reconstruction_losses.avg,
                },
            )
            model_ops.save_model(
                model,
                global_step,
                os.path.join(config["save_path"], "saved_models"),
                optimizer=optimizer,
            )

        if global_step % config["test_step"] == 0:
            dctrain_tests(
                model,
                global_step,
                config,
                jsonsaver,
                test_denoise_loader,
                test_std_loaders,
                device,
                loss_cls,
                tb_logger,
            )
            model = model.to(device)
            model.train()

        # validation
        if (global_step % config["val_step"]) == 0:
            if config["denoise_training"]:
                val_vis_loss, val_bpp_loss, val_combined_loss = tests.test_dir(
                    model=model,
                    step=global_step,
                    jsonsaver=jsonsaver,
                    config=config,
                    device="cpu" if config["test_on_cpu"] else device,
                    prefix="val_denoise",
                    loader=val_loader,
                    tb_logger=tb_logger,
                    loss_cls=loss_cls,
                    gt_src="tuple",
                    incl_combined_loss=True,
                )
            elif config["std_training"]:
                val_vis_loss, val_bpp_loss, val_combined_loss = tests.test_dir(
                    model=model,
                    step=global_step,
                    jsonsaver=jsonsaver,
                    config=config,
                    device="cpu" if config["test_on_cpu"] else device,
                    prefix="val_std",
                    loader=val_loader,
                    tb_logger=tb_logger,
                    loss_cls=loss_cls,
                    gt_src="tuple",
                    incl_combined_loss=True,
                    crop_to_multiple_size=64,
                )
            else:
                raise ValueError("Nothing to validate")

            model = model.to(device)
            model.train()
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
        if (global_step % config["save_model_freq"]) == 0:
            cleanup_checkpoints.cleanup_checkpoints(
                expname=config["expname"], checkpoints_dir=args.checkpoints_dpath
            )

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
    model.train()

    return global_step


# TODO check that pretrain is checked against None not ''


def dctrain_tests(
    model,
    global_step,
    config,
    jsonsaver,
    test_denoise_loader,
    test_std_loaders,
    device,
    loss_cls=None,
    tb_logger=None,
    prefix="test_denoise",
):
    """
    Test a model during (or like in) DC training. Model must be set
    to correct device and to train() afterwards.
    """
    res = dict()
    res["denoise"] = tests.test_dir(
        model=model,
        step=global_step,
        jsonsaver=jsonsaver,
        config=config,
        device="cpu" if config["test_on_cpu"] else device,
        prefix=prefix,
        loader=test_denoise_loader,
        tb_logger=tb_logger,
        loss_cls=loss_cls,
        gt_src="tuple",
        incl_combined_loss=True,
    )
    if test_std_loaders is not None:
        for test_name, test_loader in test_std_loaders.items():
            res[test_name] = tests.test_dir(
                model=model,
                step=global_step,
                jsonsaver=jsonsaver,
                config=config,
                device="cpu" if config["test_on_cpu"] else device,
                prefix=f"test_{test_name}",
                loader=test_loader,
                tb_logger=tb_logger,
                loss_cls=loss_cls,
                gt_src="input",
                incl_combined_loss=True,
                crop_to_multiple_size=64,
            )
    return res


def get_test_denoise_loader(config, incl_fpaths=False, fromList=True):
    # complete test sets
    test_denoise_dataset = dataset_torch_3.PickyWholeImageDenoiseDatasetFromList(
        csv_fpath=config["quality_whole_csv_fpath"],
        testing=True,
        min_quality=config["min_quality"],
        test_reserve=config["denoise_test_reserve"],
        max_quality=config.get("max_quality"),  # should always be None during training
        incl_fpaths=incl_fpaths,
        ground_truth_y_dpath=config.get(
            "ground_truth_y_dpath"
        ),  # for testing twomodels
    )

    test_denoise_loader = torch.utils.data.DataLoader(
        dataset=test_denoise_dataset, batch_size=1, shuffle=False, pin_memory=False
    )
    return test_denoise_loader


def train_handler(config: dict, jsonsaver, device):
    #  TODO break this up
    #  get model, step
    global_step, model = model_ops.get_step_and_loaded_model(config, device)

    # get optimizer
    if hasattr(model, "get_optimizer"):
        optimizer = model.get_optimizer()
        assert hasattr(
            model, "load_weights"
        )  #  the model will also load its own optimizer weights
    else:
        optimizer = OPTIMIZERS[config["optimizer_init"]]
        is_init_optimizer = True
        if (  # TODO deprecate (complexity w/ little benefit)
            config["optimizer_final"] is not None
            and config["optimizer_switch_step"] <= global_step
        ):
            optimizer = OPTIMIZERS[config["optimizer_final"]]
            print("optimizer: {}".format(str(optimizer)))
            is_init_optimizer = False
        optimizer = optimizer(
            model.get_parameters(lr=config["base_lr"]), lr=config["base_lr"]
        )
        # optimizer = optim.Adam(parameters, lr=base_lr)
        if config["pretrain"] is not None and not config["reset_optimizer"]:
            logger.info("loading optimizer:{}".format(config["pretrain"] + ".opt"))
            model_ops.load_model(optimizer, config["pretrain"] + ".opt", device=device)
            # if os.path.isfile(args.pretrain+'.opt.module'):
            #     optimizer = torch.load(args.pretrain+'.opt.module', map_location=device)
            if config["reset_lr"]:
                model_ops.reset_lr(optimizer, model, config["base_lr"])
    if config["freeze_autoencoder"] or (
        config["freeze_autoencoder_steps"] is not None
        and config["freeze_autoencoder_steps"] > global_step
    ):
        logger.info("Freezing autoencoder (experimental)")
    tb_logger = tensorboardX.SummaryWriter(os.path.join(config["save_path"], "events"))
    loss_cls = pt_helpers.get_lossclass(config["lossf"]).to(device)
    if hasattr(model, "loss_cls"):
        model.loss_cls = None

    steps_epoch = None

    # Datasets #

    # # Training
    # ## std comp train dataset
    if config["batch_size_std"] > 0:
        train_std_dataset = dcdatasets.StdDataset(
            config["train_std_data_dpaths"],
            config["image_size"],
            add_artificial_noise=config["add_artificial_noise"],
        )
        train_std_loader = torch.utils.data.DataLoader(
            dataset=train_std_dataset,
            batch_size=config["batch_size_std"],
            shuffle=True,
            pin_memory=device.type != "cpu",
            num_workers=min(2, config["batch_size_std"]),
            drop_last=True,
        )
    else:
        train_std_loader = None
    if config["batch_size_denoise"] > 0:
        if "nind_noisy_noisy" in config["ds_opts"]:  # when is this ever used?
            train_ds_class = dataset_torch_3.LazyNoiseDataset
        else:
            if config["add_artificial_noise"]:
                train_ds_class = dcdatasets.ArtificialNoiseNIND
            elif config["min_quality"] is not None:
                train_ds_class = dataset_torch_3.PickyDenoisingDatasetFromList
            else:
                train_ds_class = dataset_torch_3.DenoisingDataset
        train_denoise_dataset = train_ds_class(
            datadirs=config["train_denoise_data_dpaths"],
            test_reserve=config["denoise_test_reserve"],
            csv_fpaths=config["quality_crops_csv_fpaths"],
            cs=config["image_size"],
            min_quality=config["min_quality"],
        )
        train_denoise_loader = torch.utils.data.DataLoader(
            dataset=train_denoise_dataset,
            batch_size=config["batch_size_denoise"],
            shuffle=True,
            pin_memory=device.type != "cpu",
            num_workers=min(config["batch_size_denoise"], 2),
            drop_last=True,
        )
    else:
        train_denoise_loader = None

    # Test set
    _, test_std_loaders = datasets.get_val_test_loaders(None, config["test_std_dpaths"])

    # Validation set
    if config["denoise_training"]:
        # validation on a small part of the test denoise set
        val_denoise_dataset = dataset_torch_3.PickyWholeImageDenoiseDatasetFromList(
            csv_fpath=config["quality_whole_csv_fpath"],
            testing=True,
            min_quality=config["min_quality"],
            test_reserve=config["denoise_test_reserve"],
            limit=VAL_NIMGS,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_denoise_dataset, batch_size=1, shuffle=False, pin_memory=False
        )
    elif config["std_training"]:
        assert config["val_std_dpath"] is not None
        val_loader, _ = datasets.get_val_test_loaders(config["val_std_dpath"], None)

    test_denoise_loader = get_test_denoise_loader(config)

    if train_denoise_loader is None:
        steps_epoch = global_step // (
            len(train_std_dataset) // (config["batch_size_std"])
        )
    elif train_std_loader is None:
        steps_epoch = global_step // (
            len(train_denoise_dataset) // (config["batch_size_denoise"])
        )
    else:
        steps_epoch = global_step // min(
            len(train_std_dataset) // (config["batch_size_std"]),
            len(train_denoise_dataset) // (config["batch_size_denoise"]),
        )

    # Check that dataloaders contain something
    if config["batch_size_denoise"] > 0:
        assert len(train_denoise_loader) > 0
        assert len(test_denoise_loader) > 0
        assert len(val_loader) > 0
    if config["batch_size_std"] > 0:
        assert len(train_std_loader) > 0
    assert len(test_std_loaders) > 0
    for dataloader in test_std_loaders.values():
        assert len(dataloader) > 0

    # Launch training

    for data_epoch in range(steps_epoch, config["tot_epoch"]):
        # set stage
        model_ops.set_model_stage(
            model, global_step, config["stage_switches"], config["tot_steps"]
        )

        # get optimizer
        if hasattr(model, "get_optimizer"):
            optimizer = model.get_optimizer()
        elif (
            config["optimizer_final"] is not None
            and config["optimizer_switch_step"] <= global_step
        ):
            if is_init_optimizer:
                logger.info(
                    "Switching optimizer from {} to {}".format(
                        config["optimizer_init"], config["optimizer_final"]
                    )
                )
                optimizer = OPTIMIZERS[config["optimizer_final"]](
                    model.get_parameters(lr=config["base_lr"]), lr=config["base_lr"]
                )
                is_init_optimizer = False
        if global_step > config["tot_steps"]:
            logger.info("Ending at global_step={}".format(global_step))
            break
        if (
            config["freeze_autoencoder_steps"] is not None
            and model.frozen_autoencoder
            and global_step >= config["freeze_autoencoder_steps"]
        ):
            model.unfreeze_autoencoder()
            logger.info("unFreezing autoencoder")

        global_step = train(
            model=model,
            data_epoch=data_epoch,
            global_step=global_step,
            jsonsaver=jsonsaver,
            optimizer=optimizer,
            config=config,
            train_std_loader=train_std_loader,
            train_denoise_loader=train_denoise_loader,
            test_std_loaders=test_std_loaders,
            val_loader=val_loader,
            test_denoise_loader=test_denoise_loader,
            device=device,
            tb_logger=tb_logger,
            loss_cls=loss_cls,
        )
        cleanup_checkpoints.cleanup_checkpoints(expname=config["expname"])
        # save_model(model, global_step, save_path)


class Test_train(unittest.TestCase):
    """
    FIXME

    [compression]$ python -m unittest discover .
    or
    [compression]$ python -m unittest train.py
    TODO use a subset of the dataset s.t. it doesn't take an unrealistic amount of time to run
    unittest yields "ResourceWarning: unclosed file <_io.BufferedReader" which doesn't occur in
    normal runs.
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
        train_handler(vars(args), jsonsaver, device)
        self.assertTrue(
            os.path.isfile(
                os.path.join(MODELS_DPATH, "unittest_scratch", "trainres.json")
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    MODELS_DPATH, "unittest_scratch", "saved_models", "iter_5000.pth"
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
        train_handler(vars(args), jsonsaver, device)
        self.assertTrue(
            os.path.isfile(os.path.join(MODELS_DPATH, "unittest_load", "trainres.json"))
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    MODELS_DPATH, "unittest_load", "saved_models", "iter_5000.pth"
                )
            )
        )


if __name__ == "__main__":
    args = initfun.get_args(
        [dc_common_args.parser_add_arguments, parser_add_arguments],
        parser_autocomplete,
        def_config_fpaths=dc_common_args.DC_DEFCONF_FPATHS + DC_DEFTRAIN_FPATHS,
    )
    jsonsaver = initfun.get_jsonsaver(args)
    check_parameters(args)
    device = pt_helpers.get_device(args.device)
    train_handler(vars(args), jsonsaver, device)
