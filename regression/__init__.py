import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgbm
from regression.config import LGBM_01, Catboost_01

import matplotlib.pyplot as plt

import statsmodels.api as sm
from tqdm import tqdm
import pandas as pd
import numpy as np

DEL_LIST = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__file__', '__cached__', '__builtins__']

def pre_process(args, df):
    if args.mode == 'linear':
        config = LGBM_01
    elif args.mode == 'logistic':
        config = Catboost_01
    else:
        sys.exit(1)

    # param
    for col in DEL_LIST:
        try:
            del config.__dict__[col]
        except:
            pass
    params = config.__dict__
    
    # df pre-process
    df = df.drop(['fid', 'iid'], axis=1)
    X_data = df.drop(['y'], axis=1)
    y_data = df['y'].astype('float')
    
    # train/val split
    if args.mode == 'linear':
        x_train, x_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, shuffle=True, random_state=34)
    elif args.mode == 'logistic':
        x_train, x_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, shuffle=True, stratify=y_data, random_state=34)
    else:
        sys.exit(1)
    
    return x_train, x_val, y_train, y_val, params


#ADD + cov1 + cov2에 대해 회귀분석
def run_regression(row, df, y_data):
    i = row.name
    X_data = df[['constant', i, 'sex', 'covar']]
    model2 = sm.OLS(y_data, X_data)
    result2 = model2.fit()
    return result2.pvalues[i]

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

    # just debug #
    res_df = pd.read_csv(f'dataset/{args.mode}_result_manhattan.txt',delimiter = "\t",header = None)
    qc_cols = res_df[2].tolist()
    qc_cols.append('y')
    df = df[qc_cols]
    ##############

    #covar 파일 호출
    covars = pd.read_csv('dataset/covar.txt',delimiter = "\t",header = None)[[2,3]]
    df['sex'] = covars[2]
    df['covar'] = covars[3]
    df['constant'] = [1]*100

    # get list of resId
    resId = list(df.columns[0:-4])

    # apply regression to each row of the resId columns
    out = []
    if args.mode == 'linear':
        y_data = df['y'].astype('float')

        for id in resId:
            X_data = df[['constant', id, 'sex', 'covar']]
            model = sm.OLS(y_data, X_data)
            res = model.fit()
            out.append(
                res.pvalues[id]
            )

    elif args.mode == 'logistic':
        y_data = df['y'].astype('uint')
        y_data[y_data==1] = 0
        y_data[y_data==2] = 1

        for id in resId:
            X_data = df[['constant', id, 'sex', 'covar']]
            model = sm.Logit(y_data, X_data)
            res = model.fit()
            out.append(
                res.pvalues[id]
            )
    else:
        sys.exit(1)


    ans = res_df[5].tolist()
    res = np.isclose(ans[:len(out)], out, atol=1e-4)
    print(['_' if b else idx for idx, b in enumerate(res)][:50])

    # plt.plot(ans[:1000], label="ans")
    # plt.plot(out[:1000], label="pred")
    # plt.legend()
    # plt.show()

    #Comparison with plink data
    plink_result = pd.read_csv(f'dataset/linear_result_manhattan.txt',delimiter = "\t",header = None)    
    #파일에서 유전형 이름과 p-value 인덱스만 추출
    plink_result2 = plink_result[[2,5]]
    # res = pd.DataFrame({"id":resId, "pvalue":resValue})
    print(res)
    #plink데이터는 어떤 조건에 의해 몇몇 유전형들이 탈락하는데 일단 이건 구현 못해서 merge로 일치하는 부분만 확인
    merge_df = pd.merge(plink_result2, res, how="left", left_on=2, right_on="id")
    print(merge_df)
    print(np.isclose(merge_df[5], merge_df['pvalue']))


def run_model(args, df, num):
    x_train, x_val, y_train, y_val, params = pre_process(args, df)
    
    # train
    print('Starting training...')
    lgbm_train = lgbm.Dataset(x_train, y_train)
    lgbm_eval = lgbm.Dataset(x_val, y_val, reference=lgbm_train)
    gbm = lgbm.train(params, lgbm_train, num_boost_round=20, valid_sets=lgbm_eval, callbacks=[lgbm.early_stopping(stopping_rounds=5)])
    
    # save model
    print('Starting training...')
    gbm.save_model(f'{args.save_dir}/{args.mode}_model_{num}.txt')

    # predict
    print('Starting predicting...')
    y_pred = gbm.predict(x_val, num_iteration=gbm.best_iteration)

    # eval
    error = mean_absolute_error(y_val, y_pred)
    print(f'The MAE of prediction is: {error}')
    
    return error