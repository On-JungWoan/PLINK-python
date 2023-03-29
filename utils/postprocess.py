import pandas as pd
import numpy as np
from . import resultStatistics
import os
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
    print("Make data frame: in progress...")
    
    featureColumn = resultStatistics.txt_csv(f'dataset/{args.mode}_result_manhattan.txt', num_col)
    
    df_item = {'fid' : id[0], 'iid': id[1]}
    for idx, snp_binary in enumerate(bed):
        if idx == 0 or idx==784255:
            continue
        if snpid[idx] in featureColumn:
            df_item[ snpid[idx] ] = np.nan_to_num(snp_binary).astype('int')

    res_df = pd.DataFrame(df_item)
    print("Make data frame: Success!")
    return res_df