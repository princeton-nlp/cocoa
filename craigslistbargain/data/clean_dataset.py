"""

"""

import json
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from Dialogue import Dialogue


def print_event(e):
    print('print'+'-'*10)
    for i in e:
        print('agent:{}\taction:{}\t\t{}\tdata:{}'.format(i['agent'], i['action'], i['metadata'], i['data']))


parser = argparse.ArgumentParser()
parser.add_argument('--test', help='Generate relatively small dataset for test', default=False, action='store_true')
args = parser.parse_args()

if args.test:
    run_type = 'test'
else:
    run_type = 'run'

print('Run in {} mode.'.format(run_type))

file_names = ['dev-luis-post.json', 'train-luis-post.json']
raw_names = ['dev-luis-parsed.json', 'train-luis-parsed.json']

if run_type == 'test':
    out_names = ['dev-luis-post-small.json', 'train-luis-post-small.json']
    clean_names = ['dev-luis-small-clean.json', 'train-luis-small-clean.json']
else:
    out_names = ['dev-luis-post-new.json', 'train-luis-post-new.json']
    clean_names = ['dev-luis-clean2.json', 'train-luis-clean2.json']


# clean_names = ['dev-luis-clean.json', 'train-luis-clean.json']

empty_event = [{'start_time': '0', 'agent': 0, 'action': 'None',
                'data': '', 'metadata': {'intent': 'None', 'price': None}, 'time': '0'},
               {'start_time': '0', 'agent': 1, 'action': 'None',
                'data': '', 'metadata': {'intent': 'None', 'price': None}, 'time': '0'}]

def get_action(e):
    if e['action'] == 'message':
        return e['metadata']['intent']
    else:
        return e['action']


def add_one(d, k):
    if k not in d:
        d[k] = 1
    else:
        d[k] += 1


def check_price(e):
    if e['metadata'] is None:
        md = e['data']
    else:
        md = e['metadata']
    if md is not None and 'price' in md and md['price'] is not None:
        return True
    return False


def draw_bar(lengths):
    cc = collections.Counter(lengths)
    y = [cc[i] for i in lengths]
    plt.bar(lengths, y)
    # plt.show()


def convert_data(input, output):
    a_p = {}
    a = {}
    with open(input, 'r') as f:
        data = json.loads(f.read())

    if run_type == 'test':
        data = data[:200]
    newData = []

    lengths = []

    for i, d in enumerate(data):
        last_agent = None
        events = []
        ill = False
        for j, e in enumerate(d['events']):
            # if e['agent'] == last_agent:
            #     print(i, j)
            #     print_event(d['events'])
            #     break
            # last_agent = e['agent']
            act = get_action(e)
            if check_price(e):
                add_one(a_p, act)
            else:
                add_one(a, act)
                # if act in ['offer']:
                #     print_event(d['events'])

            # if act in ['None', 'unknown']:
            #     print(e)
            if e['metadata'] is None:
                price = None
                if (e['data'] is not None) and (e['data'].get('price') is not None):
                    price = e['data']['price']
                e['metadata'] = {'intent': act, 'price': price}
            assert e['metadata'] is not None

            # Add (dis)agree -no price
            if act in ['agree', 'disagree'] and not check_price(e):
                e['metadata']['intent'] = act+'-noprice'
                if act == 'disagree':
                    e['metadata']['intent'] = 'counter-noprice'

            if act == 'propose' and not check_price(e):
                e['metadata']['intent'] = 'inform'

            # Change original None to unknown
            if act in ['None']:
                e['action'] = 'unknown'

            if (act in ['unknown', 'None']) or \
                    (act == 'offer' and e['metadata']['price'] is None):
                ill = True
                break

            # Add None for Nothing to do
            if last_agent == e['agent']:
                events.append(empty_event[e['agent'] ^ 1])
            events.append(e)

            # Update last agent
            last_agent = e['agent']


        d['events'] = events
        if not ill:
            newData.append(d)
            lengths.append(len(events))


    draw_bar(lengths)
    data = newData

    if output is not None:
        print("file name: {}\nprice_act: {}\nact:{}".format(input, a_p, a))
        with open(output, 'w') as f:
                f.write(json.dumps(data))
    else:
        print("file name: {}\nprice_act: {}\nact:{}".format(input, a_p, a))


def _check_price_one_dialogue(d):
    d = Dialogue(d)
    last_price = [2, 2]
    history_price = []

    violate_num = [0]*10
    counter = [0]*10

    # ?: the target price of seller is the same as real price
    violate_num[0] += d.real_price != d.targets[d.seller]

    for i, e in enumerate(d.events):
        m = e['metadata']
        a = e['agent']
        if i > 0:
            # ?: people always give price lower than before
            # violate_num[5] += d.history_price[i-1][a] < d.history_price[i][a]
            #
            # if m.get('intent') == 'agree-noprice' and d.events[i-1]['metadata'].get('price') is None:
            #     violate_num[6] += 1
            pass

        if m.get('price') is None:
            continue
        p = m['price']

        # ?: people can give a price which is larger than real price
        # violate_num[1] += p > d.real_price

        # ?: people can give a price which is lower than target of buyer
        # violate_num[2] += p < d.price_lines[0]

        # ?: people can give a price which is lower than 10% of real price
        violate_num[3] += p < d.real_price*0.1

        # ?: people can give a price which is larger than 10 times of real price
        violate_num[4] += p > d.real_price * 10


    # if np.sum(violate_num[3]) > 0:
    # if violate_num[3]+violate_num[4] ==0 and violate_num[5] > 0:
    if np.sum(violate_num[1]) > 0:
        print('----------------one error')
        print(violate_num)
        strs = d.to_str()
        for i in strs:
            print(i)

    return violate_num, d

def check_for_price(fname, save_at):

    with open(fname, 'r') as f:
        data = json.loads(f.read())

    new_data = []

    counter = [0]*10
    counter = np.array(counter)

    outcome_counter = np.array([0]*10)

    for d in data:
        array, dd = _check_price_one_dialogue(d)
        array = np.array(array)
        array[array>0]=1
        # counter += np.sum(array) > 0
        counter += array
        counter[-1] += np.sum(array) > 0
        if counter[-1] > 0:
            new_data.append(d)

        outcome_counter[dd.outcome] += 1

    print('outcome:{}'.format(outcome_counter))
    print('vio:{}/{}'.format(counter, len(data)))
    print('end of check price in {}.'.format(fname))

    if save_at is not None:
        print('new dataset save at {}.'.format(save_at))
        with open(save_at, 'w') as f:
                f.write(json.dumps(new_data))

if __name__ == "__main__":
    need_check_intent = True
    need_check_price = True

    if need_check_intent:
        # Convert all the data
        print('-'*6+' Convert all the data')
        for i, f in enumerate(file_names):
            # convert_data(f, out_names[i])
            convert_data(f, out_names[i])

        # Check new data
        print('-'*6+' Check new data')
        for i, f in enumerate(out_names):
            # convert_data(f, out_names[i])
            convert_data(f, None)

    if need_check_price:
        print('-'*6+' Check all the price in data')
        for i, f in enumerate(out_names):
            check_for_price(f, clean_names[i])


