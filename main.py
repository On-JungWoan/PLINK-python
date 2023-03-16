from os.path import join
from pandas_plink import read_plink, read_plink1_bin, get_data_folder
import sys
import argparse
from utils.etc import merge_col, read_vcf
from regression import linear_regression, logistic_regression

def get_args_parser():
    parser = argparse.ArgumentParser('Default Argument', add_help=False)

    # dataset parameters
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--mode', default='logistic')

    return parser

def main(args):
    bim, fam, bed = read_plink(join(args.path, 'dataset/data')) #bim, fam, bed
    # fam = read_vcf(join(args.path, 'dataset/data.vcf'))
    genotype_df = merge_col(
        fam, join(args.path, 'dataset/sex_info.txt'), 'sex'
    )  

    if args.mode == 'linear':
        train_test_df = merge_col(
            genotype_df, join(args.path, f'dataset/{args.mode}_pheno.txt'), 'y'
        )
        print(f"**bim file**\n{bim}\n")
        print(f"**fam file**\n{fam}")
        # print(linear_regression(train_test_df))
    elif args.mode == 'logistic':
        train_test_df = merge_col(
            genotype_df, join(args.path, f'dataset/{args.mode}_pheno.txt'), 'y'
        )
        print(f"**bim file**\n{bim}\n")
        print(f"**fam file**\n{fam}")        
        # print(logistic_regression(train_test_df))
    else:
        sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

import pickle