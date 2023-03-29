import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plink 실행 시 결과로 출력된 result.txt파일을 csv형식으로 변환
#결과파일 linear_result_manhattan.txt, logistic_result_manhattan.txt
def txt_csv(result, num):
    with open(result, 'r') as f:
        lines = f.readlines()
    # 각 행을 데이터프레임의 열로 변환
    data = pd.DataFrame([line.strip().split() for line in lines])
    result = select_feature(data, num)
    return result

#pvalue를 기반으로 독립변수 개수를 5, 10, 20, 30 으로 축소
def select_feature(df, num):
    # input : plink 결과 데이터 output: 선택된 변수set
    # 데이터프레임 출력
    df.columns = ["CHR","POS","SNP","ALT","STAT","P"]
    df=df.drop(df[df['P'] == "NA"].index,axis = 0)
    manhattan(df)
    train_feature = df.sort_values("P",ascending = True).iloc[0:num,:]['SNP']
    return list(train_feature)

#결과를 기반으로 맨해튼플롯 작성
def manhattan(df):
    df2 = pd.DataFrame({'SNP':df['SNP'],'P':df['P'],'CHR':df['CHR']})
    # -log_10(pvalue)
    df2['minuslog10pvalue'] = -np.log10(df.P.astype('float'))
    df2.CHR = df2.CHR.astype('int')
    df2 = df2.sort_values('CHR')
    print(df2)
    # How to plot gene vs. -log10(pvalue) and colour it by chromosome?
    df2['ind'] = range(len(df2))
    df_grouped = df2.groupby(('CHR'))

    # manhattan plot
    fig = plt.figure(figsize=(14, 8)) # Set the figure size
    ax = fig.add_subplot(111)
    colors = ['darkred','darkgreen','darkblue', 'gold']
    x_labels = []
    x_labels_pos = []
    for num, (name, group) in enumerate(df_grouped):
        group.plot(kind='scatter', x='ind', y='minuslog10pvalue',color=colors[num % len(colors)], ax=ax)
        x_labels.append(name)
        x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0])/2))
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)

    # set axis limits
    ax.set_xlim([0, len(df)])
    ax.set_ylim([0, 5])

    # x axis label
    ax.set_xlabel('CHR')

    # show the graph
