import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgbm
from regression.config import LGBM_01, Catboost_01

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

def run_model(args, df):
    x_train, x_val, y_train, y_val, params = pre_process(args, df)
    
    # train
    print('Starting training...')
    lgbm_train = lgbm.Dataset(x_train, y_train)
    lgbm_eval = lgbm.Dataset(x_val, y_val, reference=lgbm_train)
    gbm = lgbm.train(params, lgbm_train, num_boost_round=20, valid_sets=lgbm_eval, callbacks=[lgbm.early_stopping(stopping_rounds=5)])
    
    # save model
    print('Starting training...')
    gbm.save_model(f'{args.save_dir}/{args.mode}_model.txt')

    # predict
    print('Starting predicting...')
    y_pred = gbm.predict(x_val, num_iteration=gbm.best_iteration)

    # eval
    error = mean_absolute_error(y_val, y_pred)
    print(f'The MAE of prediction is: {error}')
    
    return error