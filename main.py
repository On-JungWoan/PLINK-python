import sys
import argparse
from os.path import join
from pandas_plink import read_plink, read_plink1_bin, get_data_folder

from regression import linear_regression, logistic_regression
from utils.dataset import load_vcf
from utils.postprocess import merge_col, make_genotype_by_bed

def get_args_parser():
    parser = argparse.ArgumentParser('Default Argument', add_help=False)
    # dataset parameters
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--mode', default='linear')
    parser.add_argument('--file', default='bim')

    return parser

def make_xy(args, df):

    # pheno
    train_test_df = merge_col(
        df, join(args.path, f'dataset/{args.mode}_pheno.txt'), 'y'
    )

    return train_test_df.drop(['i'], axis=1)

def main(args):
    bim, fam, bed = read_plink(join(args.path, 'dataset/data')) #bim, fam, bed
    fid = fam['fid'].tolist()
    snpid = bim['snp'].tolist()

    # debug
    make_genotype_by_bed(bed.compute(), fid, snpid)

    if args.file == 'fam':
        # add sex_info
        genotype_df = merge_col(
            fam, join(args.path, 'dataset/sex_info.txt'), 'sex'
        )
        # add covar
        genotype_df = merge_col(
            genotype_df, join(args.path, f'dataset/covar.txt'), 'cov'
        )        

    train_test_df = make_xy(args, genotype_df)
    # print(f"**bim file**\n{bim}\n")
    # print(f"**fam file**\n{fam}")

    if args.mode == 'linear':
        print(linear_regression(args, train_test_df))
    elif args.mode == 'logistic':
        print(logistic_regression(args, train_test_df))
    else:
        sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)