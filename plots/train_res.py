import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import sys

alias = {
        'convnexttiny': 'ConvNeXt-T',
        'mobilenetv3small': 'MobileNetV3-S',
        'efficientnet': 'EfficientNetB0',
        'efficientnetv2': 'EfficientNetV2B0',
        'inceptionv3': 'InceptionV3'
        }

ys = {
    'val_categorical_accuracy': 'Categorical accuracy (%)',
    'val_top_k_categorical_accuracy': 'Categorical top-5 accuracy (%)',
    'val_loss': 'Validation Loss' if len(sys.argv) < 3 else sys.argv[-1],
}

ms = {
    'val_categorical_accuracy': 100,
    'val_top_k_categorical_accuracy': 100,
    'val_loss': 1
}

def find_and_parse(files):
    parse = lambda x: x.split('/')[-1].split('.')[0].split('_')[0]
    match = lambda x: re.match(sys.argv[2],x)
    return map(lambda x: (x, parse(x)),filter(match, files))

def data_and_aliases(data_files):
    return map(lambda x: (pd.read_csv(x[0]), alias[x[1]]), data_files)

def field(data, field,mul):
    return map(lambda x: ((x[0][field]*mul).tolist() if x[1] != 'ConvNeXt-T' else (x[0][field][-8:]*mul).tolist(), x[1]), data)

def plot(data, ax):
    return map(lambda x: ax.plot(x[0], label=x[1]), data)

def plot_some_field(ax, y, data, mul=1):
    [x for x in plot(field(data,y,mul), ax)]
    ax.set_ylabel(ys[y])
    ax.legend()
    ax.set_xlabel("Epoch")


#if __name__ == '__main__':
#    fig, axs = plt.subplots(nrows=3)
#    fig.set_figheight(10)
#    fig.subplots_adjust(hspace=0)
#    axs[0].set_title(sys.argv[3])
#    axs[-1].set_xlabel('Epochs')
#    data = list(data_and_aliases(find_and_parse(map(lambda x: sys.argv[1] + '/' + x,os.listdir(sys.argv[1])))))
#    for y,ax in zip(ys.keys(), axs):
#        plot_some_field(ax, y, data,mul=ms[y])
#    plt.show()
for y in ys.keys():
    plt.figure()
    ax = plt.subplot()
    plot_some_field(ax, y, data_and_aliases(find_and_parse(map(lambda x: sys.argv[1] + '/' + x,os.listdir(sys.argv[1])))),mul=ms[y])
plt.show()
#.*improve\.log
#[a-z0-9/]+(?<!improve.log)\.log(?<!improve.log)
