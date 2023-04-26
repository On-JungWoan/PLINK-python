import numpy as np
import pandas as pd
from tqdm import tqdm
from . import resultStatistics

def merge_col(origin_df, path, col_name):
    with open(path, 'r') as f:
        texts = f.readlines()

    id_list = []; res_list = []

    for text in texts:
        text = text.replace('\n', '')
        res = text.split('\t')
        id_list.append(res[0])
        res_list.append(res[-1])
    df = pd.DataFrame({
        'id': id_list,
        col_name: res_list
    })
    res_df = pd.merge(
        left=origin_df, right=df, how='left', left_on='fid', right_on='id'
    ).drop(['id'], axis=1)

    return res_df


def make_genotype_by_bed(args, bed, id, snpid, num_col):
    print("\nMake data frame: in progress...\n")
    
    if args.full_dataset:
        target = bed
        col = snpid
    else:
        featureColumn = resultStatistics.txt_csv(f'dataset/{args.mode}_result_manhattan.txt', num_col)
        inter_col = set(snpid).intersection(set(featureColumn))
        
        col_idx = []
        col = []
        for i_col in tqdm(inter_col):
            tmp = snpid.index(i_col)
            col_idx.append(tmp)
            col.append(snpid[tmp])
        target = bed[col_idx]

    df_item = {'fid' : id[0], 'iid': id[1]}
    for idx, snp_binary in enumerate(tqdm(target)):
        df_item[ col[idx] ] = np.nan_to_num(snp_binary).astype('int')    

    res_df = pd.DataFrame(df_item)
    print("\nMake data frame: Success!\n")
    return res_df