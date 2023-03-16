from statsmodels.formula.api import logit
import statsmodels.api as sm
from datetime import datetime
import pickle
import sys

#전처리함수
def pre_process(args, data):
    NoIdData = data.drop(['fid','iid'], axis=1)#id가 담긴 데이터들 제거
    if args.mode == 'linear':
        NoIdData['y'] = NoIdData['y'].astype(float)
    elif args.mode == 'logistic':
        NoIdData = NoIdData.replace({'y' : 1}, 0)
        NoIdData = NoIdData.replace({'y' : 2}, 1)
        NoIdData['y'] = NoIdData['y'].astype(str)
    else:
        sys.exit(0)
    y_cols=['y']
    x_cols=list(set(NoIdData).difference(set(y_cols)))
    NoIdData[x_cols] = NoIdData[x_cols].astype(float)
    print(NoIdData)
    return NoIdData, x_cols, y_cols

def run_model(args, data):
    newdata ,x_cols, y_cols = pre_process(args, data)
    model_input = f"{y_cols[0]}~"+"+".join(x_cols)

    if args.mode == "linear":
        model = sm.OLS.from_formula(model_input, data=newdata)
    elif args.mode == "logistic":
        model = logit(model_input, data=newdata)

    results = model.fit()
    if not args.nosave:
        now = datetime.now().strftime('%H_%M_%S')
        with open(f'logs/{args.mode}_{now}_results.pkl', 'wb') as f:
            pickle.dump(results, f)
    return results.summary()