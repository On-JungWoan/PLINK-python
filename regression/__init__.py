from sklearn.linear_model import LinearRegression, LogisticRegression
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import sys
from utils.decorators import logging_time
from scipy import stats

#전처리함수
def pre_process(args, data):
    NoIdData = data.drop(['fid','iid'], axis=1)#id가 담긴 데이터들 제거
    if args.mode == 'linear':
        NoIdData['y'] = NoIdData['y'].astype(np.float16)
    elif args.mode == 'logistic':
        NoIdData = NoIdData.replace({'y' : 1}, 0)
        NoIdData = NoIdData.replace({'y' : 2}, 1)
        NoIdData['y'] = NoIdData['y'].astype(str)
    else:
        sys.exit(0)
    return NoIdData

@logging_time
def run_model(args, data):
    newdata = pre_process(args, data)
#     model_input = f"{y_cols[0]}~"+"+".join(x_cols)
    print("Modeling start..")
    if args.mode == "linear":
        model = LinearRegression()
    elif args.mode == "logistic":
        model = LogisticRegression()
    results = model.fit(newdata.drop('y', axis=1),newdata['y'])

    print("Get statistics..")
    get_statistics(results, newdata.drop('y', axis=1), newdata['y'])
    if not args.nosave:
        now = datetime.now().strftime('%H_%M_%S')
        with open(f'logs/{args.mode}_{now}_results.pkl', 'wb') as f:
            pickle.dump(results, f)
    return results

@logging_time
def get_statistics(model, x, y):
    params = np.append(model.intercept_,model.coef_)
    predictions = model.predict(x)
    newX = pd.DataFrame({"Constant":np.ones(len(x))}).join(pd.DataFrame(x))
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))
    print(newX)

    # 나눌 부분행렬의 개수 지정
    num_splits = 7195
    # array_split 함수를 이용하여 부분행렬 생성 후, 각 부분행렬에서 표준편차 계산
    stds = []
    for sub_arr in np.array_split(newX, num_splits, axis=1):
        stds.append(np.std(sub_arr, axis=0))

    # 계산된 표준편차들을 다시 병합
    stds = np.concatenate(stds)
    std_error = stds / np.sqrt(newX.shape[0])
    t_values = params / std_error
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX.columns) - 1))) for i in t_values]

    std_error = np.round(std_error, 3)
    t_values = np.round(t_values, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    statistics = pd.DataFrame()
    statistics["Coefficients"], statistics["Standard Errors"], statistics["t -values"], statistics["p-values"] = [params, std_error, t_values, p_values]
    print(statistics)
    return statistics