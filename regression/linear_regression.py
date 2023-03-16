import argparse
import ast
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import statsmodels.api as sm

def linear_regression(data):
    # args 에 위의 내용 저장
    NoIdData = data.drop(['fid','iid'], axis=1)
    y_col=['y']
    x_cols = list(set(NoIdData).difference(set(y_col)))
    model = sm.OLS.from_formula(f"{y_col[0]}~"+"+".join(x_cols), data=NoIdData)
    results = model.fit(method = "newton")
    return results.summary
        # 다변수함수에 뉴턴방법을 적용한 로지스
