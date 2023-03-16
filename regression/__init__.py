
import numpy as np
import statsmodels.api as sm

def pre_process(data):
    # args 에 위의 내용 저장
    NoIdData = data.drop(['fid','iid'], axis=1)
    feature_columns = NoIdData.columns.difference(["y"])
    y_train=NoIdData['y']
    x_train=NoIdData[feature_columns]

    return x_train, y_train

def logistic_regression(data):
    x_train, y_train = pre_process(data)
    model = sm.Logit(y_train, x_train)
    results = model.fit(method = "newton")
    return results.summary