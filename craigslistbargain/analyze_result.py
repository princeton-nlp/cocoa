import numpy as np
import re
r = re.compile('([-+]?\d+\.\d+)')
with open('result.txt','r') as f:
    data = f.readlines()
rewards = [[], []]
for d in data:
    if d.find('reward: [0]') != -1:
        rewards[0].append(float(r.findall(d)[0]))
    if d.find('reward: [1]') != -1:
        rewards[1].append(float(r.findall(d)[0]))

print('rewards: {}, {}'.format(np.mean(rewards[0]), np.mean(rewards[1])))
