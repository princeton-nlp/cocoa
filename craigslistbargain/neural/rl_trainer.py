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
from neural.trainer import Trainer
from .utterance import UtteranceBuilder


class RLTrainer(Trainer):
    def __init__(self, agents, scenarios, train_loss, optim, training_agent=0, reward_func='margin', cuda=False):
        self.agents = agents
        self.scenarios = scenarios

        self.training_agent = training_agent
        self.model = agents[training_agent].env.model
        self.critic_model = agents[training_agent].env.critic_model
        self.train_loss = train_loss
        self.optim = optim
        self.optim = optim
        self.cuda = cuda

        self.best_valid_reward = None

        self.all_rewards = [[], []]
        self.reward_func = reward_func

    def update_critic(self, reward, model, discount=0.95):
        model.train()

    def update(self, batch_iter, reward, model, discount=0.95):
        model.train()
        model.generator.train()
        # if self.critic_model is not None:
        #     self.critic_model.train()

        nll = []
        # batch_iter gives a dialogue
        dec_state = None
        for batch in batch_iter:
            if not model.stateful:
                dec_state = None
            enc_state = dec_state.hidden if dec_state is not None else None

            print("batch: \nencoder{}\ndecoder{}\ntitle{}\ndesc{}".format(batch.encoder_inputs.shape, batch.decoder_inputs.shape, batch.title_inputs.shape, batch.desc_inputs.shape))
            if enc_state is not None:
                print("state: {}".format(batch, enc_state[0].shape))

            outputs, _, dec_state = self._run_batch(batch, None, enc_state)  # (seq_len, batch_size, rnn_size)
            loss, _ = self.train_loss.compute_loss(batch.targets, outputs)  # (seq_len, batch_size)
            nll.append(loss)

            # Don't backprop fully.
            if dec_state is not None:
                dec_state.detach()

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

    def validate_critic(self, args):
        split = 'dev'
        # Set model in validating mode.
        self.critic_model.eval()

        stats = Statistics()

        num_val_batches = next(valid_iter)
        dec_state = None
        for batch in valid_iter:
            if batch is None:
                dec_state = None
                continue
            elif not self.model.stateful:
                dec_state = None
            enc_state = dec_state.hidden if dec_state is not None else None

            outputs, attns, dec_state = self._run_batch(batch, None, enc_state)
            _, batch_stats = self.valid_loss.compute_loss(batch.targets, outputs)
            stats.update(batch_stats)

        # Set model back to training mode
        self.model.train()

        self.critic_model.train()
        return stats

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

                #if not args.only_run:
                self.update(batch_iter, s_reward, self.model, discount=args.discount_factor)

                #if
                self.update_critic(s_reward, self.critic_model, discount=args.discount_factor)

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

            # TODO: Save critic model
            if (i > 0 and i % 100 == 0) and not args.only_run:
                valid_stats = self.validate_critic(args)
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
