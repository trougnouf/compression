# denoise noisydirs, creating appropriate subdirs
orig_dir=$(pwd)
cd ../nind_denoise
MODELPATH='models/2019-08-03T16:14_nn_train.py_--g_network_UNet_--weight_SSIM_1_--batch_size_65_--test_reserve_ursulines-red_stefantiek_ursulines-building_MuseeL-Bobo_CourtineDeVillersDebris_MuseeL-Bobo-C500D_--train_data_datasets-train-NIND_128_96_--g_model_path_models-20/generator_280.pt'

NOISYDIRSTRAIN=( \
'../../datasets/train/resized/1024/FeaturedPictures' \
'../../datasets/train/resized/1024/Formerly_featured_pictures_on_Wikimedia_Commons' \
)
for noisydir in "${NOISYDIRSTRAIN[@]}"
do
    python denoise_dir.py --skip_existing --max_subpixels 25165824 --result_dir make_subdirs --noisy_dir ${noisydir} --model_path ${MODELPATH} --cs 512 --ucs 384 --network UNet --pad 128 --whole_image --cuda_device -1
done

NOISYDIRSTEST=( \
'../../datasets/test/clic_test_pro' \
'../../datasets/test/clic_valid_2020' \
)
for noisydir in "${NOISYDIRSTEST[@]}"
do
    python denoise_dir.py --skip_existing --max_subpixels 25165824 --result_dir make_subdirs --noisy_dir ${noisydir} --model_path ${MODELPATH} --cs 512 --ucs 384 --network UNet
done

cd ${orig_dir}