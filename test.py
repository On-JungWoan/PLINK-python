'''
preprocess함수가 변경됐기 따로 함수 구성
df: preprocess를 거치지 않은 데이터 프레임(fid, iid 제거과정 + y값 astype 따로 진행)
ADD: 유전형데이터 1개
cov1: sex데이터
cov2: covar데이터
m: 상수항
회귀식: y = b1*ADD+b2*COV1+b3*COV2+m
'''
import statsmodels.api as sm
import pandas as pd
import numpy as np

df = pd.read_csv('dataset/logistic_result_manhattan.txt', delimiter = "\t", header = None)


df = df.drop(['fid', 'iid'], axis=1)

#covar 파일 호출
covars = pd.read_csv('dataset/covar.txt',delimiter = "\t",header = None)[[2,3]]    
df['sex'] = covars[2]
df['covar'] = covars[3]
df['constraint'] = [1]*100 

#ADD + cov1 + cov2에 대해 회귀분석
def run_regression(row, df, y_data):
    i = row.name
    X_data = df[['constraint', i, 'sex', 'covar']]
    model2 = sm.OLS(y_data, X_data)
    result2 = model2.fit()
    return result2.pvalues[i]
# convert 'y' column to float
y_data = df['y'].astype('float')
# get list of resId
resId = list(df.columns[0:-4])
# apply regression to each row of the resId columns
resValue = df[resId].apply(run_regression, df=df, y_data=y_data, axis=0)

#Comparison with plink data'
plink_result = pd.read_csv(f'dataset/linear_result_manhattan.txt',delimiter = "\t",header = None)    
#파일에서 유전형 이름과 p-value 인덱스만 추출
plink_result2 = plink_result[[2,5]]
res = pd.DataFrame({"id":resId, "pvalue":resValue})
print(res)
#plink데이터는 어떤 조건에 의해 몇몇 유전형들이 탈락하는데 일단 이건 구현 못해서 merge로 일치하는 부분만 확인
merge_df = pd.merge(plink_result2, res, how="left", left_on=2, right_on="id")
print(merge_df)
print(np.isclose(merge_df[5], merge_df['pvalue']))