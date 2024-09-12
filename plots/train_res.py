import pandas as pd
import matplotlib.pyplot as plt
import os
import re

alias = {
        'convnexttiny': 'ConvNeXt-T',
        'mobilenetv3small': 'MobileNetV3-S',
        'efficientnet': 'EfficientNetB0',
        'efficientnetv2': 'EfficientNetV2B0',
        'inceptionv3': 'InceptionV3'
        }


def find_and_parse(files):
    parse = lambda x: x.split('.')[0].split('_')[0]
    match = lambda x: re.match(r'.*improve.*\.log',x)
    return map(lambda x: (x, parse(x)),filter(match, files))

def aliases_and_data(data_files):
    return map(lambda x: (pd.read_csv(x[0]), alias[x[1]]), data_files)

def field(data, field):
    return map(lambda x: (x[0][field], x[1]), data)

def plot(data):
    return map(lambda x: plt.plot(x[0], label=x[1]), data)

[x for x in plot(field(aliases_and_data(find_and_parse(os.listdir(sys.argv[1]))),"val_categorical_accuracy"))]
plt.legend()
plt.show()
