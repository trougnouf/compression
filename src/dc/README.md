# On the Importance of Denoising when Learning to Compress Images

Reference implementation in PyTorch, forking Manypriors and nind_denoise

All python code is expected to be run from within one of themain  project sub-directories (dc, compression, nind_denoise). The code specific to the joint denoising and compression project is located in `src/dc/`; the python interpreter is expected to run from there.

## Dependencies:

###Python pip:

```bash
pip3 install cffi
pip3 install python-pytorch-msssim tqdm matplotlib scipy scikit-image scikit-video ConfigArgParse pyyaml h5py ptflops colorspacious pypng piqa opencv-python jpegtran-cffi mwclient piexif
```

###Arch Linux:

```bash
sudo pacman -S python-tqdm python-pytorch-cuda python-matplotlib python-configargparse python-yaml
pacaur -S python-ptflops python-colorspacious python-pytorch-msssim-git python-jpegtran-cffi python-scikit-image
```

###Slurm:

```bash
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install cffi # needs to be installed before jpegtran-cffi, per https://github.com/jbaiter/jpegtran-cffi/issues/27
pip3 install tqdm matplotlib scipy scikit-image scikit-video ConfigArgParse pyyaml h5py ptflops colorspacious pypng piqa opencv-python jpegtran-cffi mwclient piexif
```

## Dataset gathering:

Run the following code to gather both the clean and noise datasets, and crop them:

```bash
# NIND
cd ../nind_denoise
python3 dl_ds_1.py --target_dir ../../datasets/NIND --use_wget
python crop_ds.py --cs 256 --ucs 96 --dsdir ../../datasets/NIND --resdir ../../datasets/resized --max_threads 8
# dataset of clean images
cd "../nind_denoise"
bash tools/make_clean-clean_dataset.sh
```

if the denoising dataset has been updated, then run the following code to analyze the dataset and generate the crops/quality list:

```
python3 tools/make_dataset_quality_crops_list.py --train_data ../../datasets/cropped/NIND_256_192 --cs 256
```

to generate ../../datasets/cropped/msssim.csv (which should then be renamed to msssim.csv and adapted in the quality_crops_csv_fpaths configuration key)

For testing, the file ../../datasets/NIND-msssim.csv is needed. It can be generated with `nind_denoise$ python tools/make_dataset_quality_images_list.py `

Add the 24-images kodak test set to ../../datasets/test/kodak/

To train with the JDC-UD method (aka pre-denoise or knowledge distillation), prepare the training data as follow:

```
cd ../nind_denoise
python denoise_dir.py --model_path ../../models/nind_denoise/2021-06-14T20:27_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_with_clean_data.yaml_--g_model_path_..-..-models-nind_denoise-2021-06-12T11:48_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_w/generator_650.pt --batch_size 2 --noisy_dir '../../datasets/train/FeaturedPictures' --skip_existing
```

To test denoising-then-compression, prepare the test data as follow:

```
cd ../nind_denoise
python denoise_dir.py --model_path ../..//models/nind_denoise/2021-06-14T20:27_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_with_clean_data.yaml_--g_model_path_..-..-models-nind_denoise-2021-06-12T11:48_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_w/generator_650.pt --batch_size 2 --noisy_dir '../../datasets/test/kodak' --skip_existing --denoise_baselines_too

python denoise_dir.py --model_path ../../models/nind_denoise/2021-06-14T20:27_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_with_clean_data.yaml_--g_model_path_..-..-models-nind_denoise-2021-06-12T11:48_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_w/generator_650.pt --batch_size 2 --noisy_dir '../../datasets/test/clic_test_pro' --skip_existing --denoise_baselines_too

```

# Train

`python3 train.py` starts a training process from scratch. The provided configurations can be used with `--config config/dctrain_<method>.yaml`. Start with the default `--train_lambda 4096` (highest bitrate) and use this model to train the following (lower bitrate) model (e.g. `--train_lambda 2048`). When resuming training at a lower bitrate, it is necessary to reset the learning rate with `--reset_lr`.

Egcommand to train a `JDC-Cn.8` model with λ=4096: `python3 train.py --config dctrain_lownoise80_nind_fp.yaml --train_lambda 4096 --tot_steps 6000000`

Egcommand to train a `JDC-UD` model with λ=4096: `python3 train.py --config dctrain_predenoised8_nind_fp.yaml --train_lambda 4096`

## Loading a trained model

Use `--pretrain` where "pretrain" is the name of the directory under `../../models/dc/` (e.g. `dctrain_lownoise80_nind_fp-L4096-mb-cas001.cism.ucl.ac.be`. The full path of the model is not required; it will be loaded according to the best validation loss (`val_denoise_combined_loss`) present in the file `trainres.json`.

Egcommand to train a `JDC-Cn.8` model with λ=2048: `python3 train.py --config dctrain_lownoise80_nind_fp.yaml --train_lambda 2048 --pretrain dctrain_lownoise80_nind_fp-L4096-mb-cas001.cism.ucl.ac.be --reset_lr --tot_steps 9000000`

## Miscellaneous

To pause a training process, create a `~/pause_<PID>` file. This will pause GPU compute but RAM usage will remain.

# Test

- Add the models to test in a config file such as `config/models_to_test.yaml`, using the following keys:
  - `models_rootdir`: `dc` for jdc models, `compression` for standard compression models
  - `model_name`: trained model directory name, as in the `pretrain` argument. eg: `dctrain_lownoise80_nind_fp-L4096-mb-cas001.cism.ucl.ac.be`
  - `train_method`: `lownoise` for JDC-Cn, `predenoise` for JDC-UD, `allnoise` for JDC-CN, `onlynoise` for JDC-N, `twomodels` for a standard method preceded by a universal denoiser, `std` for a standard learned method, `null` for a standard codec
  - `train_lambda`: eg: 4096 and below. Use 0 for standard codec
  - `lossf`: loss function by which the model was finally trained (mse or msssim), used to compute the "combined_loss"
- launch `python3 tools/make_res.py` to test models, optionally with the `--list_of_models_yamlpath` and/or `--list_of_tests_yamlpath` arguments.
- launch `python3 tools/graph_res.py` to graph all known results; use a `--config` (s.a. the examples in config/graph_*.yaml) to generate specific graphs.

The models test performance are saved in `../../models/dc/<MODELNAME>/trainres.json`. Likewise, individual images' results, both a cache of the scores and encoded/decoded images, are saved under `../../models/dc/<MODELNAME>/tests`
