
# Load scalar from tensorboard logs and render with matplotlib
#   * extract_rewards_from_example(file_name):
#       Get rewards from specific examples file
#   * load_from_tensorboard
#       Get logs (include rewards) from tensorboard

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import re

edge = 0.17
left_e = 0.19
up_e = 0.92
right_e = 0.98

def extract_rewards_from_example(file_name='result.txt'):
    r = re.compile('([-+]?\d+\.\d+)')
    with open(file_name,'r') as f:
        data = f.readlines()
    rewards = [[], []]
    for d in data:
        if d.find('reward: [0]') != -1:
            rewards[0].append(float(r.findall(d)[0]))
        if d.find('reward: [1]') != -1:
            rewards[1].append(float(r.findall(d)[0]))
    return [np.mean(rewards[0]), np.mean(rewards[1])]
    # print('rewards: {}, {}'.format(np.mean(rewards[0]), np.mean(rewards[1])))

def get_args():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dir', default='logs/', help='directory of the tensorboard log file')
    parser.add_argument('--draw-type', choices=['sl', 'rl', 'tom'], type=str, default='sl', help='plot type')
    parser.add_argument('--show', action='store_true', help='show pictures directly')
    parser.add_argument('--fig-dir', default='./', help='direcotry of these figures')
    parser.add_argument('--font-size', type=int, default=24, help='size of fonts')
    parser.add_argument('--label-size', type=int, default=24, help='size of font on label')

    parser.add_argument('--test', type=str, nargs='+', choices=['a', 'b', 'c'], help='size of font on label')

    args = parser.parse_args()
    print(args.test)
    quit()
    return args

def load_from_tensorboard(dir):
    data = []

    # for root, dirs, files in os.walk(dir):
    #     for file in files:
    #         fpath = os.path.join(root, file)
    ea = event_accumulator.EventAccumulator(dir)
    ea.Reload()
    # print(ea.scalars.Keys())

    # val_psnr = ea.scalars.Items('val_psnr')
    # print(len(val_psnr))
    # print([(i.step, i.value) for i in val_psnr])
    return ea.scalars

def get_xy(it):
    x, y = [], []
    for i in it:
        x.append(i.step)
        y.append(i.value)
    return x, y

# Supervise Learning
# dev/train loss
def draw_sl_training_curve(dir, args):
    sl_scalars = load_from_tensorboard(dir)

    # Draw text
    labels = ['train set', 'dev set']
    x = []
    y = []
    plt.style.use('ggplot')

    # Draw train set
    x, y = get_xy(sl_scalars.Items('train/loss'))
    plt.plot(x, y, label=labels[0])

    # Draw dev set
    x, y = get_xy(sl_scalars.Items('dev/loss'))
    plt.plot(x, y, label=labels[1])

    plt.xlabel('Traning steps', fontsize=args.font_size)
    plt.ylabel('Loss', fontsize=args.font_size)
    plt.title('Suprevise Learning', fontsize=args.font_size)
    plt.tick_params(labelsize=args.label_size)
    # plt.subplots_adjust(bottom=edge, left=left_e, top=up_e, right=right_e)
    plt.legend(fontsize=args.font_size)
    if args.show:
        plt.show()
    else:
        plt.savefig()

# def draw_it_var(max_var=2):
#     #f = open('record/bias%.2f.pkl' % max_var, 'rb')
#     f = open('record/var_%.1f.pkl'%max_var, 'rb')
#     loss_h = pickle.load(f)
#     # color = ['magenta', 'orange', 'green', 'red']
#     if max_var == 2:
#         plt.figure(figsize=(6.5,5))
#     else:
#         plt.figure(figsize=(6.3,5))
#     for i in range(4):
#         if i == 1:
#             plt.plot([])
#             plt.fill_between([],[],[])
#             continue
#         #loss_h[i] = loss_h[i][:70]
#         y = [np.mean(np.array(j)) for j in loss_h[i]]
#         x = list(range(len(y)))
#         std = [np.std(np.array(j)) for j in loss_h[i]]
#         sup = [y[j] + std[j] for j in range(len(y))]
#         inf = [y[j] - std[j] for j in range(len(y))]
#
#         sup = [y[j] + std[j] for j in range(len(y))]
#         inf = [y[j] - std[j] for j in range(len(y))]
#
#         plt.plot(x, y, label=labels[i], )
#         plt.fill_between(x, inf, sup, alpha=0.3)
#         plt.ylim((0, y[0]*0.4))
#         plt.xlim((0, len(x)-10))
#
#     plt.xlabel('Budget', fontsize=font_size)
#     plt.ylabel('Loss', fontsize=font_size)
#     #plt.title('Variance in [0.1, %.1f]' % (max_var), fontsize=font_size)
#     plt.tick_params(labelsize=labelsize)
#     plt.subplots_adjust(bottom=edge, left = left_e, top=up_e, right=right_e)
#     plt.legend(fontsize=font_size)
#     plt.show()

def draw_rl_tranning_curve(dir, args):
    rl_scalars = load_from_tensorboard(dir)

    # Draw text
    labels = ['RL Agent', 'SL Agent']
    x = []
    y = []
    plt.style.use('ggplot')
    plt.figure(figsize=(6.3, 5))

    # Draw train set
    x, y = get_xy(rl_scalars.Items('agent0/reward'))
    plt.plot(x, y, label=labels[0])

    # Draw dev set
    x, y = get_xy(rl_scalars.Items('agent1/reward'))
    plt.plot(x, y, label=labels[1])

    plt.xlabel('Traning steps', fontsize=args.font_size)
    plt.ylabel('Reward', fontsize=args.font_size)
    plt.title('Reinforcement Learning', fontsize=args.font_size)
    plt.tick_params(labelsize=args.label_size)
    plt.subplots_adjust(bottom=edge, left=left_e, top=up_e, right=right_e)
    plt.legend(fontsize=args.font_size)
    if args.show:
        plt.show()
    else:
        plt.savefig()

if __name__ == '__main__':
    args = get_args()
    # data = load_from_tensorboard(args.dir)

    if args.draw_type == 'sl':
        # Draw sl curve
        draw_sl_training_curve(args.dir, args)
    elif args.draw_type == 'rl':
        # Draw rl curve
        draw_rl_tranning_curve(args.dir, args)
    elif args.draw_type == 'tom':
        pass


