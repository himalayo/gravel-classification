import sys
import pathlib
import numpy as np
import os
import json
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def generate_confusion_matrix(matrix, columns, indexes):
    #sns.set(rc={'figure.figsize': (27, 5)})  # Size in inches
    #columns = [f"feature{i}" for i in range(0,27)]
    #indexes = ['AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'GS', 'HD',  'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT']
    df = pd.DataFrame(matrix, columns=columns, index=indexes)
    df.index.name = 'Actual'

    df_long = df.reset_index().melt(id_vars='Actual', var_name='Expected', value_name='%')
    sns.set_style('darkgrid')
    g = sns.relplot(data=df_long, x="Actual", y="Expected", size="%", hue="%",
                    marker="s", sizes=(20, 200), palette="blend:limegreen,orange", height=8, aspect=1.1)
    g.ax.tick_params(axis='x', labelrotation=45)
    g.ax.set_facecolor('aliceblue')
    g.ax.grid(color='red', lw=1)

    g.fig.subplots_adjust(left=0.1, bottom=0.15)
    plt.show()

if len(sys.argv) <= 2:
    print(f'Usage: {sys.argv[0]} <dataset_path> <model>')

titles = {
            'mobilenetv3small': 'MobileNetv3-S',
            'convnexttiny': 'ConvNeXt-T',
            'inceptionv3': 'InceptionV3',
            'efficientnet': 'EfficientNetB0',
            'efficientnetv2': 'EfficientNetV2B0'
        }


with open(sys.argv[1]) as subdirs:
    classes = list(filter(lambda x: x!='overall',json.load(subdirs).keys()))

for file in map(open, sys.argv[2:]):
    data = json.load(file)
    file.close()
    fig, ax = plt.subplots(figsize=(9,8),gridspec_kw={'left':0.23,'bottom':0.25,'top':0.95,})
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)


    ax.title.set_text(titles[pathlib.Path(file.name).stem.split('_')[0]])
    epitaph = np.array(next(filter(lambda x: x['label']=='overall', data))['epitaph'])
    new_epitaph = []
    for i,row in enumerate(epitaph):
        div = np.copy(1/np.sum(epitaph[i,:]))
        new_epitaph.append(row*div)
    epitaph = np.array(new_epitaph)*100
    #generate_confusion_matrix(epitaph, classes, classes)

    im = ax.imshow(epitaph,cmap='Blues',vmin=0,vmax=100, aspect='auto')
    for i,row in enumerate(epitaph):
        for j,elem in enumerate(row):
            color = im.cmap(0.9) if elem < 50 else im.cmap(0)
            ax.text(j,i,f'{elem:.0f}',color=color,ha='center',va='center',fontsize='medium',fontweight='bold')


    cbar = ax.figure.colorbar(im, ax=ax,fraction=0.046)
    ax.set_xticks(range(27), labels = classes)
    ax.set_yticks(range(27), labels = classes)
    plt.setp(ax.get_xticklabels(),rotation=70,ha='right',rotation_mode='anchor')
    ax.set_yticks(np.arange(28)-.5, minor=True)

    ax.tick_params(which='minor',bottom=False,left=False,labelsize='x-small')
    ax.tick_params(which='major',labelsize='small')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, *cbar.ax.get_yticklabels()] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)

    fig.savefig(pathlib.Path(file.name).stem.split('_')[0]+'.pdf')


matplotlib.rcParams.update({'font.size': 22})
matplotlib.rc('axes',linewidth=8)
plt.show()
