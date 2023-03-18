from sklearn.linear_model import LinearRegression, LogisticRegression
from datetime import datetime
import numpy as np
import pickle
import sys

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

def run_model(args, data):
    newdata = pre_process(args, data)
#     model_input = f"{y_cols[0]}~"+"+".join(x_cols)
    print("make model")
    if args.mode == "linear":
        model = LinearRegression()
    elif args.mode == "logistic":
        model = LogisticRegression()
    results = model.fit(newdata.drop('y', axis=1),newdata['y'])
    if not args.nosave:
        now = datetime.now().strftime('%H_%M_%S')
        with open(f'logs/{args.mode}_{now}_results.pkl', 'wb') as f:
            pickle.dump(results, f)
    return results


def get_statistics(model, intercept=0, coef=[], train_df=None, target_df=None):
    print("Coefficient :", coef)
    print("Intercept :", intercept)
    
    params = np.append(intercept, coef)
    print("Params:", params)

    #prediction = model.predict(train_df.values.reshape(-1, 1))        # 단변량
    prediction = model.predict(train_df.values)                     # 다변량

    if len(prediction.shape) == 1:
        prediction = np.expand_dims(prediction, axis=1)
    print(train_df.columns)

    new_trainset = pd.DataFrame({"Constant": np.ones(len(train_df.values))},dtype='float16').join(pd.DataFrame(train_df.values))
    print(new_trainset)
    
    from sklearn.metrics import mean_squared_error
    MSE = mean_squared_error(prediction, target_df.values)
    print("MSE :", MSE)
    new_trainset = new_trainset.astype('float16')
    variance = MSE * (np.linalg.inv(np.dot(new_trainset.T, new_trainset)).diagonal())       # MSE = (1, ) & else = (n, ) 가 나와야 함.

    std_error = np.sqrt(variance)
    t_values = params / std_error
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(new_trainset) - len(new_trainset.columns) - 1))) for i in t_values]

    std_error = np.round(std_error, 3)
    t_values = np.round(t_values, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    statistics = pd.DataFrame()
    statistics["Coefficients"], statistics["Standard Errors"], statistics["t -values"], statistics["p-values"] = [params, std_error, t_values, p_values]

    return statistics