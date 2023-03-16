import pandas as pd
 
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

def make_genotype_by_bed(bed, bim):
    print("Make data frame: in progress...")
    snp_name = bim['snp'].tolist()

    df_item = {}
    for idx, snp_binary in enumerate(bed):
        if idx == 0 or idx==784255:
            continue

        df_item[ snp_name[idx] ] = snp_binary

    res_df = pd.DataFrame(df_item)
    print()