import json
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from Dialogue import Dialogue

# from .analyzer_config import

file_names = ['dev-luis-post.json', 'train-luis-post.json']
raw_names = ['dev-luis-parsed.json', 'train-luis-parsed.json']

parser = argparse.ArgumentParser()
parser.add_argument('--test', help='Generate relatively small dataset for test', default=False, action='store_true')
args = parser.parse_args()

if args.test:
    run_type = 'test'
else:
    run_type = 'run'

if run_type == 'test':
    out_names = ['dev-luis-post-small.json', 'train-luis-post-small.json']
    clean_names = ['dev-luis-small-clean.json', 'train-luis-small-clean.json']
else:
    out_names = ['dev-luis-post-new.json', 'train-luis-post-new.json']
    clean_names = ['dev-luis-clean2.json', 'train-luis-clean2.json']


def output_strange_dialogue(d):
    return clean_names


def only_changed(price, change, d):
    x = []
    y = []
    for i, p in enumerate(price):
        if change[i]:
            # p = d.transfer_price(p, d.events[i]['agent'], d.seller)
            p = d.price_to_normal(p, d.events[i]['agent'])
            x.append(i)
            y.append(p)
    return x, y

def reshape_x(xx):
    idx = []
    for i in range(2):
        for j in xx[i]:
            idx.append(j)
    idx.sort()
    dct = {}
    for i, j in enumerate(idx):
        dct[j] = i
    xxx = [[], []]
    for i in range(2):
        for j in xx[i]:
            xxx[i].append(dct[j])
    return xxx


if __name__ == "__main__":
    ds = Dialogue.load_from_file(file_names[0])

    for j, d in enumerate(ds):
        ps = list(zip(*d.history_price))
        ch = list(zip(*d.price_changed))
        xx = []
        yy = []
        not_empty = False
        for i in range(2):
            x, y = only_changed(ps[i], ch[i], d)
            xx.append(x)
            yy.append(y)
            if len(x) > 0:
                not_empty = True

        xx = reshape_x(xx)

        print('[Dialogue] {}/{}'.format(j, len(ds)))
        # for s in d.to_str():
        #     print(s)
        if not_empty:
            # print(ps[i], ch[i])
            color = ['blue', 'orange']
            color = [None]*2
            for i in range(2):
                x, y = xx[i], yy[i]
                plt.plot(x, y, marker='o', color=color[i])
                # print(x, y)
            plt.ylim(0.0, 2.5)
            plt.savefig('./pic/{}.png'.format(j))
            plt.close('all')
        else:
            # print('is empty!')
            pass

    # plt.show()