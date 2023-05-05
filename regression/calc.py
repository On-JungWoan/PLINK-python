import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm

#ADD + cov1 + cov2에 대해 회귀분석
def run_regression(model, row, df):
    X_data = df[['constant', row, 'sex', 'covar']]
    y_data = df['y']
       
    m = model(y_data, X_data)
    res = m.fit(disp=0, warn_convergence=False, method='bfgs')
    return res.pvalues[row]
        
    
def createPvalue(args, df, bim):
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

    #covar 파일 호출
    covars = pd.read_csv('dataset/covar.txt',delimiter = "\t",header = None)[[2,3]]
    df['sex'] = covars[2]
    df['covar'] = covars[3]
    df['constant'] = [1]*100

    # get list of resId
    resId = list(df.columns[0:-4])
    
    # calc p-value
    out = []
    if args.mode == 'linear':
        df['y'] = df['y'].astype(np.float32)
        for id in tqdm(resId):
            out.append(
                run_regression(sm.OLS, id, df)
            )
            
    elif args.mode == 'logistic':
        df['y'] = df['y'].astype(np.int32)
        df['y'] = df['y'].replace([1,2], [0,1])

        for id in tqdm(resId):
            out.append(
                run_regression(sm.Logit, id, df)
            )
    else:
        sys.exit(1)

    bim.index = bim['snp']
    res_df = bim.loc[resId,:][['chrom', 'snp']]
    res_df['p'] = out

    return res_df