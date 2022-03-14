'''
List the NIND dataset (whole images under ../../datasets/NIND) and its MS-SSIM score compared to
baseline(s)
'''
import configargparse
import os
import sys
sys.path.append('..')
from nind_denoise import dataset_torch_3
from nind_denoise import nn_common
#  from nind_denoise import nn_train
from common.libs import utilities
from common.libs import libimganalysis

DATASETS_DPATHS = [os.path.join('..', '..', 'datasets', 'NIND')]

if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description=__doc__, config_file_parser_class=configargparse.YAMLConfigFileParser)
    #  , default_config_files=[nn_common.COMMON   _CONFIG_FPATH, nn_train.DEFAULT_CONFIG_FPATH],

    parser.add('-c', '--config', is_config_file=True, help='(yaml) config file path')
    parser.add_argument('--dataset_dpaths', nargs='*', default=DATASETS_DPATHS, help="(space-separated) Path(s) to the noise dataset data")
    parser.add_argument('--debug_options', '--debug', nargs='*', default=[], help=f"(space-separated) Debug options (available: {nn_common.DebugOptions})")
    args, _ = parser.parse_known_args()
    debug_options = [nn_common.DebugOptions(opt) for opt in args.debug_options]

    results = list()

    if len(args.dataset_dpaths) == 1:
        output_fpath = args.dataset_dpaths[0]+'-msssim.csv'
    elif len(args.dataset_dpaths) > 1:
        output_fpath = '_'.join([utilities.get_leaf(apath) for apath in args.dataset_dpaths])+'-msssim.csv'
    else:
        raise ValueError(args.dataset_dpaths)

    for ds_dpath in args.dataset_dpaths:
        for aset in os.listdir(ds_dpath):
            print(f'{aset=}')
            set_dpath = os.path.join(ds_dpath, aset)
            iso_fn_dict = {fn.split('_')[-1].split('.')[0]: fn for fn in os.listdir(set_dpath)}
            try:
                bisos, nisos = dataset_torch_3.sort_ISOs(iso_fn_dict.keys())
            except ValueError as e:
                print(f'make_dataset_quality_images_list.py error: {e}. Hint: check that {set_dpath} is not empty?')
                breakpoint()
            print(f'{bisos=}, {nisos=}')
            for biso in bisos:
                for niso in nisos:
                    bisopath = os.path.join(set_dpath, iso_fn_dict[biso])
                    nisopath = os.path.join(set_dpath, iso_fn_dict[niso])
                    score = libimganalysis.piqa_msssim(bisopath, nisopath)
                    print({'xpath': bisopath, 'ypath': nisopath, 'score': score})
                    results.append((bisopath, nisopath, score))

    # save
    utilities.list_of_tuples_to_csv(results, ('xpath', 'ypath', 'score'), output_fpath)
    print(f'Quality check exported to {output_fpath}')
