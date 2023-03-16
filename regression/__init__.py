from statsmodels.formula.api import logit
import statsmodels.api as sm
import pandas as pd
#전처리함수
def pre_process(data):
    NoIdData = data.drop(['fid','iid'], axis=1)#id가 담긴 데이터들 제거
    NoIdData['y'] = NoIdData['y'].astype(int)
    NoIdData = NoIdData.replace({'y' : 1}, 0)
    NoIdData = NoIdData.replace({'y' : 2}, 1)
    NoIdData['father'] = NoIdData['father'].astype(int)
    NoIdData['mother'] = NoIdData['mother'].astype(int)
    NoIdData['gender'] = NoIdData['gender'].astype(int)
    NoIdData['i'] = NoIdData['i'].astype(int)
    NoIdData['sex'] = NoIdData['sex'].astype(int)
    NoIdData['trait'] = NoIdData['trait'].astype(int)
    y_cols=['y']
    x_cols=list(set(NoIdData).difference(set(y_cols)))
    print(NoIdData)
    return NoIdData, x_cols, y_cols

def logistic_regression(data):
    newdata ,x_cols, y_cols = pre_process(data)
    model = logit(f"{y_cols[0]}~"+"+".join(x_cols), data=newdata)
    results = model.fit()
    return results.summary
        # 다변수함수에 뉴턴방법을 적용한 로지스틱

def linear_regression(data):
    newdata ,x_cols, y_cols = pre_process(data)
    model = sm.OLS.from_formula(f"{y_cols[0]}~"+"+".join(x_cols), data=newdata)
    results = model.fit()
    return results.summary