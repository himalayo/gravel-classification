#!/usr/bin/env python3

import pathlib
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

_identity = lambda x: x

class Category:
    def __init__(self, name, loss, top_1, top_5):
        self.name = name
        self.loss = loss
        self.top_1 = top_1
        self.top_5 = top_5

    @staticmethod
    def from_dict(data):
        return Category(data['label'], data['results']['loss'], data['results']['categorical_accuracy'], data['results']['top_5_accuracy'])

class Entry:
    def __init__(self, name, categories):
        self.name = name
        self.categories = categories

    @staticmethod
    def from_json(filename, name_parser = _identity, criteria = _identity):
        with open(filename) as f:
            raw_data = json.load(f)

        categories = []
        for category_dict in filter(criteria,raw_data):
            categories.append(Category.from_dict(category_dict))

        return Entry(name_parser(filename), categories)

    def top_1(self):
        for category in self.categories:
            yield category.top_1

    def top_5(self):
        for category in self.categories:
            yield category.top_5

    def loss(self):
        for category in self.categories:
            yield category.loss

    def __call__(self):
        for category in self.categories:
            yield category

    def __repr__(self):
        return f'Entry: {{ Name: {self.name}  Accuracies: {str(list(self.top_1()))} }}'

    def __str__(self):
        return self.__repr__()

class EntryList:
    def __init__(self, entries):
        self.entries = entries

    @staticmethod
    def from_json_list(filenames, **kwargs):
        entries = []
        for filename in filenames:
            entries.append(Entry.from_json(filename, **kwargs))
        return EntryList(entries)

    def scatter(self, xs, ys,criteria = _identity, label=False):
        for entry in self.entries:
            plt.scatter(list(map(xs, entry())), list(map(ys, entry())), label=entry.name if label else None)

    def plot(self,xs, ys,criteria=_identity, label=False):
        for entry in filter(criteria,self.entries):
            unordered_xs = list(map(xs, entry()))
            unordered_ys = list(map(ys, entry()))
            order = sorted(zip(unordered_xs, unordered_ys))
            x = list(map(lambda u: u[0], order))
            y = list(map(lambda u: u[1], order))
            plt.plot(x,y, label=entry.name if label else None)

    def bar(self, xs, ys, criteria=_identity):
        up = {}
        down = {}
        for x,y in map(lambda entry: (list(map(xs, entry())), list(map(ys, entry()))),self.entries):
            for i,j in zip(x,y):
                try:
                    if up[i] < j:
                        up[i] = j
                except:
                    up[i] = j

                try:
                    if down[i] > j:
                        down[i] = j
                except:
                    down[i] = j

        height = np.array(list(up.values()))-np.array(list(down.values()))
        print(len(list(up.keys()))==len(list(up.values())))
        plt.bar(list(up.keys()),height,bottom=list(down.values()),width=4e2)


def normalizer_by_label(normalizer):
    return lambda category: category.top_1/normalizer(category.name)

def dict_fn(d):
    return lambda x: d[x]

if __name__ == '__main__':
        if len(sys.argv) < 2:
                exit()
        matplotlib.rcParams.update({'font.size': 22})
        matplotlib.rc('axes',linewidth=4)

        parse = lambda filename: filename.split('.')[0].split('_')[0]
        alias = {
            'efficientnet': 'EfficientNetB0',
            'efficientnetv2': 'EfficientNetV2',
            'convnexttiny': 'ConvNeXt-T',
            'inceptionv3': 'InceptionV3',
            'mobilenetv3small': 'MobileNetV3-S'
        }

        name_fn = lambda filename: alias[parse(filename)]

        with open(sys.argv[1]) as f:
            train_count = json.load(f)

        xs = lambda x: train_count[x.name]
        ys = lambda y: y.top_1*100

        criteria = lambda raw_cat: raw_cat['label'] != 'overall'
        entry_list = EntryList.from_json_list(sys.argv[2:], name_parser = name_fn, criteria = criteria)
        entry_list.plot(xs,ys, criteria=lambda entity: entity.name==alias['convnexttiny'], label=False)
        entry_list.scatter(xs,ys,label=True)

        plt.legend()
        plt.xlabel('Number of samples')
        plt.ylabel('Accuracy (%)')
        plt.show()
