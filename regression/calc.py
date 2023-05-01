import statsmodels.api as sm
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import sys

#ADD + cov1 + cov2에 대해 회귀분석
def run_regression(model, row, df):
    df = df[['constant', row, 'sex', 'covar', 'y']].dropna()
    X_data = df[['constant', row, 'sex', 'covar']]
    y_data = df['y']
    if df.shape[0] == 0:
        return sys.exit(1)
    else:
        m = model(y_data, X_data)
        res = m.fit()
        return res.pvalues[row]
    
def createPvalue(args, df):
    '''
    preprocess함수가 변경됐기 따로 함수 구성
    df: preprocess를 거치지 않은 데이터 프레임(fid, iid 제거과정 + y값 astype 따로 진행)
    ADD: 유전형데이터 1개
    cov1: sex데이터
    cov2: covar데이터
    m: 상수항
    회귀식: y = b1*ADD+b2*COV1+b3*COV2+m
    '''
    df = df.drop(['fid', 'iid'], axis=1)

    #### just debug ####
    res_df = pd.read_csv(f'dataset/{args.mode}_result_manhattan.txt',delimiter = "\t",header = None)
    res_df.index = res_df[2]
    # qc_cols = res_df[2].tolist()
    # qc_cols.append('y')
    # df = df[qc_cols]
    ###################

    #covar 파일 호출
    covars = pd.read_csv('dataset/covar.txt',delimiter = "\t",header = None)[[2,3]]
    df['sex'] = covars[2]
    df['covar'] = covars[3]
    df['constant'] = [1]*100

    # get list of resId
    resId = list(df.columns[0:-4])
    # 임시 코드 -> 나중에 수정 #
    resId = list(set(resId).intersection(set(res_df.index)))
    ########################    

    # calc p-value
    out = []
    if args.mode == 'linear':
        df['y'] = df['y'].astype('float')
        for id in tqdm(resId):
            out.append(
                run_regression(sm.OLS, id, df)
            )
            
    elif args.mode == 'logistic':
        df['y'] = df['y'].astype('uint')
        df['y'][df['y']==1] = 0
        df['y'][df['y']==2] = 1

        for id in tqdm(resId):
            out.append(
                run_regression(sm.Logit, id, df)
            )
    else:
        sys.exit(1)

    res_df.index = res_df[2]
    ans_df = res_df.loc[resId,:]
    ans = ans_df[5].tolist()
    res = np.isclose(ans, out, atol=1e-4)
    print(f"# of the not matched data : {len(res[res==False])}")

    # not_matched_snp = []
    # for idx, b in enumerate(res):
    #     if not b:
    #         not_matched_snp.append(ans_df.iloc[idx,2])

    # with open('not_matched_snp.pkl', 'wb') as f:
    #     pickle.dump(not_matched_snp, f)

    # print()
    # print(['_' if b else idx for idx, b in enumerate(res)][:50]) # debug

    # with open('logs/res.pkl', 'wb') as f:
    #     pickle.dump(res,f)