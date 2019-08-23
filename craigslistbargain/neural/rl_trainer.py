import argparse
import random
import json
import numpy as np
import copy
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Variable

from cocoa.neural.rl_trainer import Statistics

from core.controller import Controller
from .utterance import UtteranceBuilder

from neural.sl_trainer import SLTrainer as BaseTrainer, Statistics, SimpleLoss

import math, time, sys


class RLStatistics(BaseTrainer):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, reward=0, n_words=0):
        self.loss = loss
        self.n_words = n_words
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.reward += stat.reward


    def mean_loss(self):
        return self.loss / self.n_words

    def mean_reward(self):
        return self.loss / self.n_words

    def elapsed_time(self):
        return time.time() - self.start_time


    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def str_loss(self):
        return "loss: %6.4f reward: %6.4f;" % (self.mean_loss(), self.mean_reward())

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d;" + self.str_loss() +
               "%6.0f s elapsed") %
              (epoch, batch,  n_batches,
               time.time() - start))
        sys.stdout.flush()

class RLTrainer(BaseTrainer):
    def __init__(self, agents, scenarios, train_loss, optim, training_agent=0, reward_func='margin', cuda=False):
        self.agents = agents
        self.scenarios = scenarios

        self.training_agent = training_agent
        self.model = agents[training_agent].env.model
        self.train_loss = SimpleLoss(inp_with_sfmx=False)
        self.optim = optim
        self.cuda = cuda

        self.best_valid_reward = None

        self.all_rewards = [[], []]
        self.reward_func = reward_func

    def update(self, batch_iter, reward, model, discount=0.95):
        model.train()

        nll = []
        # batch_iter gives a dialogue
        dec_state = None
        for batch in batch_iter:

            # print("batch: \nencoder{}\ndecoder{}\ntitle{}\ndesc{}".format(batch.encoder_inputs.shape, batch.decoder_inputs.shape, batch.title_inputs.shape, batch.desc_inputs.shape))
            # if enc_state is not None:
            #     print("state: {}".format(batch, enc_state[0].shape))

            policy, price = self._run_batch(batch)  # (seq_len, batch_size, rnn_size)
            loss, batch_stats = self._compute_loss(batch, policy, price, self.train_loss)

            loss = loss.view(-1)
            nll.append(loss)

            # TODO: Don't backprop fully.
            # if dec_state is not None:
            #     dec_state.detach()

        nll = torch.cat(nll)  # (total_seq_len, batch_size)

        rewards = [Variable(torch.zeros(1, 1).fill_(reward))]
        for i in range(1, nll.size(0)):
            rewards.append(rewards[-1] * discount)
        rewards = rewards[::-1]
        rewards = torch.cat(rewards)

        if self.cuda:
            loss = nll.squeeze().dot(rewards.squeeze().cuda())
        else:
            loss = nll.squeeze().dot(rewards.squeeze())

        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 1.)
        self.optim.step()

    def _get_scenario(self, scenario_id=None, split='train'):
        scenarios = self.scenarios[split]
        if scenario_id is None:
            scenario = random.choice(scenarios)
        else:
            scenario = scenarios[scenario_id % len(scenarios)]
        return scenario

    def _get_controller(self, scenario, split='train'):
        # Randomize
        if random.random() < 0.5:
            scenario = copy.deepcopy(scenario)
            scenario.kbs = (scenario.kbs[1], scenario.kbs[0])
        sessions = [self.agents[0].new_session(0, scenario.kbs[0]),
                    self.agents[1].new_session(1, scenario.kbs[1])]
        return Controller(scenario, sessions)

    def validate(self, args):
        split = 'dev'
        self.model.eval()
        total_stats = Statistics()
        print('='*20, 'VALIDATION', '='*20)
        for scenario in self.scenarios[split][:200]:
            controller = self._get_controller(scenario, split=split)
            example = controller.simulate(args.max_turns, verbose=args.verbose)
            session = controller.sessions[self.training_agent]
            reward = self.get_reward(example, session)
            stats = Statistics(reward=reward)
            total_stats.update(stats)
        print('='*20, 'END VALIDATION', '='*20)
        self.model.train()
        return total_stats

    def save_best_checkpoint(self, checkpoint, opt, valid_stats):
        if self.best_valid_reward is None or valid_stats.mean_reward() > self.best_valid_reward:
            self.best_valid_reward = valid_stats.mean_reward()
            path = '{root}/{model}_best.pt'.format(
                        root=opt.model_path,
                        model=opt.model_filename)

            print('Save best checkpoint {path}'.format(path=path))
            torch.save(checkpoint, path)

    def checkpoint_path(self, episode, opt, stats):
        path = '{root}/{model}_reward{reward:.2f}_e{episode:d}.pt'.format(
                    root=opt.model_path,
                    model=opt.model_filename,
                    reward=stats.mean_reward(),
                    episode=episode)
        return path

    def learn(self, args):
        rewards = [None]*2
        s_rewards = [None]*2

        for i in range(args.num_dialogues):
            # Rollout
            scenario = self._get_scenario()
            controller = self._get_controller(scenario, split='train')
            example = controller.simulate(args.max_turns, verbose=args.verbose)

            for session_id, session in enumerate(controller.sessions):
                # Only train one agent
                if args.only_run != True and session_id != self.training_agent:
                    continue

                # Compute reward
                reward = self.get_reward(example, session)
                # Standardize the reward
                all_rewards = self.all_rewards[session_id]
                all_rewards.append(reward)
                s_reward = (reward - np.mean(all_rewards)) / max(1e-4, np.std(all_rewards))

                rewards[session_id] = reward
                s_rewards[session_id] = s_reward

                batch_iter = session.iter_batches()
                T = next(batch_iter)

                self.update(batch_iter, reward, self.model, discount=args.discount_factor)

            if args.verbose:
                strs = example.to_text()
                for str in strs:
                    print(str)
                print("reward: [0]{} [1]{}".format(self.all_rewards[0][-1], self.all_rewards[1][-1]))
                # print("Standard reward: [0]{} [1]{}".format(s_rewards[0], s_rewards[1]))

            if ((i + 1) % args.report_every) == 0:
                import seaborn as sns
                import matplotlib.pyplot as plt
                if args.histogram:
                    sns.set_style('darkgrid')
                for j in range(2):
                    print('agent={}'.format(j), end=' ')
                    print('step:', i, end=' ')
                    print('reward:', rewards[j], end=' ')
                    print('scaled reward:', s_rewards[j], end=' ')
                    print('mean reward:', np.mean(self.all_rewards[j]))
                    if args.histogram:
                        self.agents[j].env.dialogue_generator.get_policyHistogram()
                print('-'*10)
                if args.histogram:
                    plt.show()

            # Save model
            if (i > 0 and i % 100 == 0) and not args.only_run:
                valid_stats = self.validate(args)
                self.drop_checkpoint(args, i, valid_stats, model_opt=self.agents[self.training_agent].env.model_args)


    def _is_valid_dialogue(self, example):
        special_actions = defaultdict(int)
        for event in example.events:
            if event.action in ('offer', 'quit', 'accept', 'reject'):
                special_actions[event.action] += 1
                # Cannot repeat special action
                if special_actions[event.action] > 1:
                    print('Invalid events(0): ')
                    for x in example.events:
                        print('\t', x.action)
                    return False
                # Cannot accept or reject before offer
                if event.action in ('accept', 'reject') and special_actions['offer'] == 0:
                    print('Invalid events(1): ')
                    for x in example.events:
                        print('\t', x.action)
                    return False
        return True

    def _is_agreed(self, example):
        if example.outcome['reward'] == 0 or example.outcome['offer'] is None:
            return False
        return True

    def _margin_reward(self, example):
        # No agreement
        if not self._is_agreed(example):
            print('No agreement')
            return {'seller': -0.5, 'buyer': -0.5}

        rewards = {}
        targets = {}
        kbs = example.scenario.kbs
        for agent_id in (0, 1):
            kb = kbs[agent_id]
            targets[kb.role] = kb.target

        midpoint = (targets['seller'] + targets['buyer']) / 2.

        price = example.outcome['offer']['price']
        norm_factor = abs(midpoint - targets['seller'])
        rewards['seller'] = (price - midpoint) / norm_factor
        # Zero sum
        rewards['buyer'] = -1. * rewards['seller']
        return rewards

    def _length_reward(self, example):
        # No agreement
        if not self._is_agreed(example):
            print('No agreement')
            return {'seller': -0.5, 'buyer': -0.5}

        # Encourage long dialogue
        rewards = {}
        for role in ('buyer', 'seller'):
            rewards[role] = len(example.events) / 10.
        return rewards

    def _fair_reward(self, example):
        # No agreement
        if not self._is_agreed(example):
            print('No agreement')
            return {'seller': -0.5, 'buyer': -0.5}

        rewards = {}
        margin_rewards = self._margin_reward(example)
        for role in ('buyer', 'seller'):
            rewards[role] = -1. * abs(margin_rewards[role]) + 2.
        return rewards


    def _balance_reward(self, example):
        # No agreement
        if not self._is_agreed(example):
            print('No agreement')
            return {'seller': -0.5, 'buyer': -0.5}

        rewards = {}
        targets = {}
        kbs = example.scenario.kbs
        for agent_id in (0, 1):
            kb = kbs[agent_id]
            targets[kb.role] = kb.target

        midpoint = (targets['seller'] + targets['buyer']) / 2.

        price = example.outcome['offer']['price']
        norm_factor = abs(midpoint - targets['seller'])
        rewards['seller'] = (price - midpoint) / norm_factor
        # Zero sum
        rewards['buyer'] = -1. * rewards['seller']

        for role in ('buyer', 'seller'):
            rewards[role] += len(example.events) / 20.

        return rewards


    def get_reward(self, example, session):
        if not self._is_valid_dialogue(example):
            print('Invalid')
            rewards = {'seller': -2., 'buyer': -2.}
        else:
            if self.reward_func == 'margin':
                rewards = self._margin_reward(example)
            elif self.reward_func == 'fair':
                rewards = self._fair_reward(example)
            elif self.reward_func == 'length':
                rewards = self._length_reward(example)
            elif self.reward_func == 'balance':
                rewards = self._balance_reward(example)
        reward = rewards[session.kb.role]
        return reward
