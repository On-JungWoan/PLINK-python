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
    return NoIdData, x_cols, y_cols

def run_model(args, data):
    newdata ,x_cols, y_cols = pre_process(args, data)
    model_input = f"{y_cols[0]}~"+"+".join(x_cols)
    print(model_input)
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



# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from datetime import datetime
# import pickle
# import sys

# #pytorch로 선형, 로지스틱 회귀분석 torch pip
# class LinearRegression(nn.Module):
#     def __init__(self, input_size, output_size): 
#         super(LinearRegression, self).__init__()
#         self.linear = nn.Linear(input_size, output_size)

#     def forward(self, x):
#         out = self.linear(x)
#         return out

# class LogisticRegression(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(LogisticRegression, self).__init__() 
#         self.linear = nn.Linear(input_size, output_size)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         return self.sigmoid(self.linear(x))
    
# def pre_process(args, data):
#     NoIdData = data.drop(['fid','iid'], axis=1)#id가 담긴 데이터들 제거
#     NoIdData = NoIdData.head()
#     if args.mode == 'linear':
#         NoIdData['y'] = NoIdData['y'].astype(np.float16)
#     elif args.mode == 'logistic':
#         NoIdData = NoIdData.replace({'y':1}, 0)
#         NoIdData = NoIdData.replace({'y':2}, 1)
#         NoIdData['y'] = NoIdData['y'].astype(np.int8)
#     else:
#         sys.exit(0)
#     return NoIdData

# def run_model(args, data):
#     newdata = pre_process(args, data)
#     train_features = torch.FloatTensor(newdata.drop('y', axis=1).values)
#     train_target = torch.FloatTensor(newdata['y'].values)
#     if args.mode == "logistic":
#         model = LogisticRegression(train_features.shape[1], 1)
#         criterion = nn.CrossEntropyLoss()

#     elif args.mode == "linear":
#         model = LinearRegression(train_features.shape[1],1)
#         criterion = nn.MSELoss()
#     # 최적화 알고리즘 정의
#     optimizer = optim.SGD(model.parameters(), lr=0.01)
#     # 학습 반복문
#     num_epochs = 1000
#     for epoch in range(num_epochs):
#     # 예측값 계산
#         train_pred = model(train_features)
#     # 손실 계산
#         train_loss = criterion(train_pred.squeeze(), train_target)
#     # 역전파 및 가중치 갱신
#         optimizer.zero_grad()
#         train_loss.backward()
#         optimizer.step()
#     # 로그 출력
#     if(epoch+1) % 100== 0:
#         print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss.item()))

#     if not args.nosave:
#         now = datetime.now().strftime('%H_%M_%S')
#     with open(f'logs/{args.mode}_{now}_results.pkl', 'wb') as f:
#         pickle.dump(train_pred, f)
#     return train_pred