import pickle
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join
import matplotlib.pyplot as plt

from regression import createPvalue
# from utils.dataset import load_vcf
from utils.resultStatistics import manhattan
from utils.postprocess import merge_col, make_genotype_by_bed
from pandas_plink import read_plink, read_plink1_bin, get_data_folder
#qc library
from utils.qualitycontrol import qc_process


def get_args_parser():
    parser = argparse.ArgumentParser('Default Argument', add_help=False)

    # dataset parameters
    parser.add_argument('--data_path', type=str, default='dataset')
    parser.add_argument('--full_dataset', default=False, action='store_true')
    parser.add_argument('--num_col', default=[5000, 10000, 15000], nargs='+')

    # regression
    parser.add_argument('--mode', default='logistic')

    # manhattan
    parser.add_argument('--save_dir', default='output')
    parser.add_argument('--no_manhattan', default=False, action='store_true')

    return parser


def make_xy(args, df):
    # pheno
    train_test_df = merge_col(
        df, join(args.data_path, f'{args.mode}_pheno.txt'), 'y'
    )
    return train_test_df


def main(args):
    bim, fam, bed = read_plink(join(args.data_path, 'data')) #bim, fam, bed


    full_sample_id = [fam['fid'].tolist(), fam['iid'].tolist()]
    full_snpid = bim['snp'].tolist()
    for num in tqdm(args.num_col):
        vcf_df = make_genotype_by_bed(bed, args.full_dataset, num, full_snpid, full_sample_id)
        # drop na
        vcf_df.replace([-9223372036854775808],[np.nan],inplace=True)
        vcf_df.dropna(axis=1, inplace=True)
        vcf_df = qc_process(vcf_df)
        train_test_df = make_xy(args, vcf_df)
        result_df = createPvalue(args, train_test_df, bim)
        
        if not args.no_manhattan:
            manhattan(args, result_df, num)

        if args.full_dataset:
            break



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)