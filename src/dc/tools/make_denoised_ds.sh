# denoise fp images
cd ../nind_denoise
python3 denoise_dir.py --no_scoring --model_path ../../models/nind_denoise/2021-06-14T20:27_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_with_clean_data.yaml_--g_model_path_..-..-models-nind_denoise-2021-06-12T11:48_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_w/generator_650.pt --batch_size 2 --noisy_dir '../../datasets/train/FeaturedPictures' --skip_existing
# crop denoised fp images
cd ../common
python3 tools/crop_ds.py --cs 1024 --ds_dir train/denoised/2021-06-14T20:27_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_with_clean_data.yaml_--g_model_path_..-..-models-nind_denoise-2021-06-12T11:48_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_w/FeaturedPictures
# denoise NIND images
cd ../nind_denoise
python3 tools/dl_ds_1.py --use_wget  # dl NIND
python3 denoise_dir.py --denoise_baselines_too --model_path ../../models/nind_denoise/2021-06-14T20:27_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_with_clean_data.yaml_--g_model_path_..-..-models-nind_denoise-2021-06-12T11:48_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_w/generator_650.pt --batch_size 2 --noisy_dir '../../datasets/NIND' --skip_existing --keep_subdir_struct
# crop denoised nind images
python3 tools/crop_ds.py --cs 256 --stride 192 --dsdir ../../datasets/denoised/2021-06-14T20:27_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_with_clean_data.yaml_--g_model_path_..-..-models-nind_denoise-2021-06-12T11:48_nn_train.py_--config_configs-train_conf_utnet_std.yaml_--config2_configs-train_w/NIND
cd ../dc
python tools/compare_imagedirs.py
