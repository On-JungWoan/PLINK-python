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


def make_genotype_by_bed(bed, full_dataset, num_col, full_snpid, full_sample_id):
    # make vcf format dataframe
    print("\n[Make data frame] in progress...")
    vcf_df = pd.DataFrame(
        bed.compute().transpose(1,0),
        index=full_sample_id[0],
        columns=full_snpid
    )
    vcf_df['fid'] = full_sample_id[0]
    vcf_df['iid'] = full_sample_id[1]

    # pre-processing
    vcf_df.replace([-9223372036854775808],[np.nan],inplace=True)
    print("[Make data frame] Success!")

    if not full_dataset:
        col = np.random.choice(full_snpid, num_col)
        return vcf_df[[*col, 'fid', 'iid']]

    return vcf_df