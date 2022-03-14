DESTDIR=/orb/tmp/cleanrepos/
ORIGDIR="$(pwd)/../.."
CMPDIR="$(pwd)"
mkdir -p ${DESTDIR}
cd ${DESTDIR}
#git clone git@github.com:trougnouf/compression.git
DESTDIR=${DESTDIR}/compression
cd ${DESTDIR}
#git pull
#rm ${DESTDIR}/* -r
cd ${CMPDIR}
mkdir -p ${DESTDIR}/datasets/test/
mkdir -p ${DESTDIR}/src/compression/libs
mkdir -p ${DESTDIR}/src/compression/tools
mkdir -p ${DESTDIR}/src/compression/models
mkdir -p ${DESTDIR}/src/compression/checkpoints/mse_4096_manypriors_64pr/saved_models
mkdir -p ${DESTDIR}/src/compression/checkpoints/mse_4096_hyperprior/saved_models
mkdir -p ${DESTDIR}/src/compression/checkpoints/mse_4096_1pr/saved_models
mkdir -p ${DESTDIR}/src/common/tools
mkdir -p ${DESTDIR}/src/common/libs
mkdir -p ${DESTDIR}/src/common/extlibs/glasbey
mkdir -p ${DESTDIR}/src/compression/config
mkdir -p ${DESTDIR}/datasets/test/Commons_Test_Photographs
mkdir -p ${DESTDIR}/src/dc
mkdir -p ${DESTDIR}/src/dc/config
mkdir -p ${DESTDIR}/src/dc/libs
mkdir -p ${DESTDIR}/src/dc/models
mkdir -p ${DESTDIR}/src/dc/resources
mkdir -p ${DESTDIR}/src/dc/scripts
mkdir -p ${DESTDIR}/src/dc/tools
mkdir -p ${DESTDIR}/src/nind_denoise
mkdir -p ${DESTDIR}/src/nind_denoise/libs
mkdir -p ${DESTDIR}/src/nind_denoise/libs/pytorch_ssim
mkdir -p ${DESTDIR}/src/nind_denoise/networks
mkdir -p ${DESTDIR}/src/nind_denoise/tool
mkdir -p ${DESTDIR}/src/nind_denoise/configs
mkdir -p ${DESTDIR}/src/nind_denoise/tools/


# std compression
cd ../compression
cp train.py tests.py ${DESTDIR}/src/compression/
cp tools/cleanup_checkpoints.py ${DESTDIR}/src/compression/tools/
cp libs/datasets.py libs/initfun.py libs/Meter.py libs/model_ops.py ${DESTDIR}/src/compression/libs/

#cp -R ../../datasets/test/Commons_Test_Photographs ${DESTDIR}/datasets/test/
#cp -R "../../datasets/test/Commons_Test_Photographs/Garden_snail_moving_down_the_Vennbahn_in_disputed_territory_(DSCF5879).png" ${DESTDIR}/datasets/test/Commons_Test_Photographs
cp ../../datasets/test/README.txt ${DESTDIR}/datasets/test/
#cp -R ../../datasets/test/kodak ${DESTDIR}/datasets/test/
#cp -r ../common/freelibs/ ${DESTDIR}/src/common/libs/
cp ../common/extlibs/pt_ms_ssim.py ../common/extlibs/radam.py ../common/extlibs/pt_rangelars.py ${DESTDIR}/src/common/extlibs/
cp models/abstract_model.py models/Balle2018PT_compressor.py models/GDN.py models/manynets_compressor.py models/bitEstimator.py models/passthrough.py models/testolina_compressor.py models/standard_compressor.py ${DESTDIR}/src/compression/models/
cp ../common/extlibs/glasbey/glasbey.py ../common/extlibs/glasbey/__init__.py ../common/extlibs/glasbey/view_palette.py ${DESTDIR}/src/common/extlibs/glasbey/
cp config/defaults.yaml ../../config/compression/mse_4096_manypriors_64pr.yaml ../../config/compression/mse_2048_manypriors_64pr.yaml ../../config/compression/mse_4096_hyperprior.yaml ${DESTDIR}/src/compression/config/
#cp checkpoints/mse_4096_manypriors_64pr/saved_models/checkpoint.pth ${DESTDIR}/src/compression/checkpoints/mse_4096_manypriors_64pr/saved_models/
cp ../common/tools/wikidownloader.py ../common/tools/verify_images.py ${DESTDIR}/src/common/tools/
cp README.md ${DESTDIR}/src/compression/

cd ../nind_denoise
cp dataset_torch_3.py denoise_dir.py denoise_image.py loss.py nn_common.py nn_train.py README.md ${DESTDIR}/src/nind_denoise/
cp configs/* ${DESTDIR}/src/nind_denoise/configs
cp libs/graph_utils.py ${DESTDIR}/src/nind_denoise/libs/
cp libs/pytorch_ssim/__init__.py ${DESTDIR}/src/nind_denoise/libs/pytorch_ssim
cp networks/ThirdPartyNets.py networks/UtNet.py networks/nnModules.py networks/p2p_networks.py ${DESTDIR}/src/nind_denoise/networks/
cp tools/* ${DESTDIR}/src/nind_denoise/tools/

cd ../dc
cp README.md ${DESTDIR}/src/dc/
cp README.md ${DESTDIR}/
cp LICENSE ${DESTDIR}/
cp train.py ${DESTDIR}/src/dc/
cd config
cp dctrain.yaml dctrain_lownoise80_nind_fp.yaml dctrain_lownoise_nind_fp.yaml dctrain_lownoise_nind_fp_msssim80.yaml dctrain_lownoise_nind_fp_msssimft.yaml dctrain_lownoise_nind_fp_testolina.yaml dctrain_predenoised8_nind_fp.yaml dctrain_predenoised9_nind_fp.yaml dctrain_testolina.yaml defaults.yaml defaults_std.yaml graph_85_onemodel.yaml graph_85_twomodels.yaml graph_add_dc.yaml graph_clicpro.yaml graph_denoisethencompress.yaml graph_hq.yaml graph_last.yaml graph_lownoise.yaml graph_lownoise_2.yaml graph_lownoise_3.yaml graph_one.yaml graph_wmsssim.yaml models_to_test.yaml test_comparisons.yaml ${DESTDIR}/src/dc/config/
cd ..
cp libs/*.py ${DESTDIR}/src/dc/libs/
cp models/*.py ${DESTDIR}/src/dc/models/
cp scripts/* ${DESTDIR}/src/dc/scripts/
cp tools/*.py tools/*.sh ${DESTDIR}/src/dc/tools/

cd ../common
cp libs/distinct_colors.py libs/json_saver.py libs/libimganalysis.py libs/np_imgops.py libs/pt_losses.py libs/pt_ops.py libs/utilities.py libs/pt_helpers.py libs/locking.py ${DESTDIR}/src/common/libs/

#cp checkpoints/mse_4096_tfcodeexp_adam/saved_models/iter_5542500.pth ${DESTDIR}/src/compression/checkpoints/mse_4096_hyperprior/saved_models/checkpoint.pth
#cp checkpoints/mse_4096_b2017_2018_swopt/saved_models/iter_3285000.pth ${DESTDIR}/src/compression/checkpoints/mse_4096_1pr/saved_models/checkpoint.pth
