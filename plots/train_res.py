import pandas as pd
import matplotlib.pyplot as plt
import os
import re

alias = [
        'convnexttiny': 'ConvNeXt-T',
        'mobilenetv3small': 'MobileNetV3-S',
        'efficientnet': 'EfficientNetB0',
        'efficientnetv2': 'EfficientNetV2B0',
        'inceptionv3': 'InceptionV3'
        ]

parse = lambda x: x.split('.')[0].split('_')[0]
match = lambda x: re.match(r'.*improve.*\.log',x)

data = []

