import sys
import pathlib
import numpy as np
import os
import json
import matplotlib.pyplot as plt

if len(sys.argv) <= 2:
    print(f'Usage: {sys.argv[0]} <dataset_path> <model>')

titles = {
            'mobilenetv3small': 'MobileNetv3-S',
            'convnexttiny': 'ConvNeXt-T',
            'inceptionv3': 'InceptionV3',
            'efficientnet': 'EfficientNetB0',
            'efficientnetv2': 'EfficientNetV2B0'
        }

classes = list(sorted(next(iter(os.walk(sys.argv[1])))[1]))

for file in map(open, sys.argv[2:]):
    data = json.load(file)
    file.close()
    fig, ax = plt.subplots(figsize=(10,9),gridspec_kw={'left':0.25,'bottom':0.3,'top':0.95})
    ax.title.set_text(titles[pathlib.Path(file.name).stem.split('_')[0]])
    epitaph = np.array(next(filter(lambda x: x['label']=='overall', data))['epitaph'])
    new_epitaph = []
    for i,row in enumerate(epitaph):
        div = np.copy(1/np.sum(epitaph[i,:]))
        new_epitaph.append(row*div)
    epitaph = np.array(new_epitaph)*100

    im = ax.imshow(epitaph,cmap='Blues',vmin=0,vmax=100)
    for i,row in enumerate(epitaph):
        for j,elem in enumerate(row):
            color = im.cmap(0.9) if elem < 50 else im.cmap(0)
            ax.text(j,i,f'{elem:.0f}',color=color,ha='center',va='center',fontsize='x-small',fontweight='bold')


    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(27), labels = classes)
    ax.set_yticks(range(27), labels = classes)
    plt.setp(ax.get_xticklabels(),rotation=60,ha='right',rotation_mode='anchor')
    ax.set_yticks(np.arange(28)-.5, minor=True)
    ax.set_xticks(np.arange(28)-.5, minor=True)
    ax.tick_params(which='minor',bottom=False,left=False,labelsize=8)
    ax.tick_params(which='major',labelsize='medium')
    fig.savefig(pathlib.Path(file.name).stem.split('_')[0]+'.pdf')


plt.show()
