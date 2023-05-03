import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#결과를 기반으로 맨해튼플롯 작성

def manhattan(args, df):
    #1. 유전형 이름 2.pvalue, 3.염색체이름
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
    ax.show()

    plt.savefig(f'dataset/{args.mode}_manhattan.png')