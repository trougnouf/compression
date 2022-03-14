cd "../nind_denoise"
python3 tools/dl_ds_1.py --use_wget  # dl NIND
python3 tools/crop_ds.py --cs 256 --stride 192  # crop NIND
bash tools/make_clean-clean_dataset.sh  # dl, filter, and crop FP
cd ../dc