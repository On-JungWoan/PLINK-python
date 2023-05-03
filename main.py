import pickle
import argparse
from tqdm import tqdm
from os.path import join
import matplotlib.pyplot as plt

from regression import createPvalue
# from utils.dataset import load_vcf
from utils.resultStatistics import manhattan
from utils.postprocess import merge_col, make_genotype_by_bed
from pandas_plink import read_plink, read_plink1_bin, get_data_folder

def get_args_parser():
    parser = argparse.ArgumentParser('Default Argument', add_help=False)
    # dataset parameters
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--mode', default='linear')
    parser.add_argument('--file', default='bim')
    parser.add_argument('--nosave', default=False, action='store_true')
    parser.add_argument('--save_dir', default='logs')

    parser.add_argument('--full_dataset', default=False, action='store_true')
    parser.add_argument('--num_col', default=[5000, 10000, 15000], nargs='+')
    #우-추
    parser.add_argument('--manhattan', default=False, action='store_true')
    return parser

def make_xy(args, df):
    # pheno
    train_test_df = merge_col(
        df, join(args.path, f'dataset/{args.mode}_pheno.txt'), 'y'
    )
    return train_test_df


def main(args):
    bim, fam, bed = read_plink(join(args.path, 'dataset/data')) #bim, fam, bed
    full_sample_id = [fam['fid'].tolist(), fam['iid'].tolist()]
    full_snpid = bim['snp'].tolist()

    for num in tqdm(args.num_col):
        vcf_df = make_genotype_by_bed(bed, args.full_dataset, num, full_snpid, full_sample_id)
        train_test_df = make_xy(args, vcf_df)
    
        result_df = createPvalue(args, train_test_df)
        
        if args.manhattan:
            manhattan(args, result_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arguments', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)