import argparse
import random
import json
import numpy as np
import copy
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Variable

from core.controller import Controller
from .utterance import UtteranceBuilder

from tensorboardX import SummaryWriter

from neural.rl_trainer import RLTrainer as BaseTrainer
from neural.sl_trainer import Statistics, SimpleLoss

import math, time, sys


class RLStatistics(Statistics):
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
        self.reward=reward
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.reward += stat.reward

    def mean_loss(self):
        return self.loss / self.n_words

    def mean_reward(self):
        return self.reward / self.n_words

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

class SimpleCriticLoss(nn.Module):
    def __init__(self):
        super(SimpleCriticLoss, self).__init__()
        self.criterion = nn.MSELoss()

    # def _get_correct_num(self, enc_policy, tgt_intents):
    #     enc_policy = enc_policy.argmax(dim=1)
    #     tmp = (enc_policy == tgt_intents).cpu().numpy()
    #     tgt = tgt_intents.data.cpu().numpy()
    #     tmp[tgt==19] = 1
    #     import numpy as np
    #     return np.sum(tmp)

    def forward(self, pred, oracle, pmask=None):
        loss = self.criterion(pred, oracle)
        stats = self._stats(loss, pred.shape[0])
        return loss, stats

    def _stats(self, loss, data_num):
        return RLStatistics(loss=loss.item(), n_words=data_num)

class RLTrainer(BaseTrainer):
    def __init__(self, agents, scenarios, train_loss, optim, training_agent=0, reward_func='margin',
                 cuda=False, args=None):
        super(RLTrainer, self).__init__(agents, scenarios, train_loss, optim,
                                        training_agent, reward_func, cuda, args)

    def _run_batch_a2c(self, batch):
        value = self._run_batch_critic(batch)
        policy, price, pvar = self._run_batch(batch)
        # print('max price', torch.max(price))
        return value, policy, price, pvar

    def _gradient_accumulation(self, batch_iter, reward, model, critic, discount=0.95):
        # Compute losses
        model.train()
        critic.train()

        values = []
        p_losses = []
        e_losses = []
        penalties = []

        # batch_iter gives a dialogue
        policy_stats = Statistics()

        for batch in batch_iter:
            # print("batch: \nencoder{}\ndecoder{}\ntitle{}\ndesc{}".format(batch.encoder_inputs.shape, batch.decoder_inputs.shape, batch.title_inputs.shape, batch.desc_inputs.shape))
            # batch.mask_last_price()
            value, policy, price, pvar = self._run_batch_a2c(batch)
            policy_loss, pl_stats = self._compute_loss(batch, policy=policy, price=(price, pvar), loss=self.train_loss)
            policy_stats.update(pl_stats)
            entropy_loss, _ = self._compute_loss(batch, policy=policy, price=(price, pvar), loss=self.entropy_loss)

            # penalty = ((price-1)**2).mul((price>2).float()) + ((price-0)**2).mul((price<0.5).float())
            # penalty = ((price > 2).float()).mul((price - 1) ** 2) + ((price < 0.5).float()).mul((price - 0) ** 2)
            penalty = ((price > 2).float()).mul(0.1) + ((price < 0.5).float()).mul(0.1)

            penalties.append(penalty.view(-1))
            p_losses.append(policy_loss.view(-1))
            e_losses.append(entropy_loss.view(-1))
            values.append(value.view(-1))

        # print('allnll ', nll)
        rewards = [0] * len(values)
        rewards[-1] = torch.ones(1) * reward

        old_rewards = [0] * len(values)
        old_rewards[-1] = torch.ones(1) * reward
        values = torch.cat(values)  # (total_seq_len, batch_size)
        for i in range(len(rewards) - 2, -1, -1):
            rewards[i] = torch.ones(1) * rewards[i]
            old_rewards[i] = old_rewards[i + 1] * discount

        # print(old_rewards)
        old_rewards = torch.cat(old_rewards)

        for i in range(len(rewards) - 2, -1, -1):
            rewards[i] += (values[i + 1].cpu().item()) * discount

        rewards = torch.cat(rewards)
        if self.cuda:
            old_rewards = old_rewards.cuda()
            rewards = rewards.cuda()

        value_loss, vl_stats = self._compute_loss(None, value=values, oracle=rewards, loss=self.critic_loss)
        # print('values', values, p_losses)
        old_p_losses = torch.cat(p_losses).view(rewards.shape)
        # p_losses = old_p_losses.mul(old_rewards).mean()
        # print('shapes', old_p_losses, old_rewards)
        # p_losses = old_p_losses.mul(old_rewards).mean()
        p_losses = old_p_losses.mul(rewards - values.detach())
        e_losses = torch.cat(e_losses)
        regular = torch.cat(penalties)
        return p_losses, e_losses, value_loss, regular, (old_p_losses, policy_stats)

    def update_a2c(self, batch_iters, rewards, model, critic, discount=0.95):
        p_losses, e_losses, value_loss, regular = None, None, None, None
        old_p_losses = None
        policy_stats = Statistics()
        for i, bi in enumerate(batch_iters):
            p,e,v,r, info = self._gradient_accumulation(bi, rewards[i], model, critic, discount)
            if p_losses is None:
                p_losses, e_losses, value_loss, regular = p,e,v,r
                old_p_losses = info[0]
            else:
                p_losses = torch.cat([p_losses, p], dim=-1)
                e_losses = torch.cat([e_losses, e], dim=-1)
                value_loss = torch.cat([value_loss, v], dim=-1)
                regular = torch.cat([regular, r], dim=-1)
                old_p_losses = torch.cat([old_p_losses, info[0]], dim=-1)
            policy_stats.update(info[1])

        # Update step
        p_losses = p_losses.mean()
        e_losses = e_losses.mean()
        value_loss = value_loss.mean()
        regular = regular.mean()

        # final_loss = p_losses - self.ent_coef * e_losses + self.val_coef * value_loss + self.p_reg_coef * regular
        # final_loss = p_losses + self.val_coef * value_loss
        final_loss = p_losses - self.ent_coef * e_losses + self.val_coef * value_loss + self.p_reg_coef * regular
        model_loss = p_losses - self.ent_coef * e_losses + self.p_reg_coef * regular
        critic_loss = self.val_coef * value_loss

        critic.zero_grad()
        # print('all loss', final_loss, p_losses, e_losses, value_loss)
        assert not torch.isnan(final_loss)
        # final_loss.backward()
        # model_loss.backward()
        # critic_loss.backward()
        # nn.utils.clip_grad_norm(critic.parameters(), 1.)
        # nn.utils.clip_grad_norm(model.parameters(), 1.)
        # self.optim.step()

        critic_loss.backward()
        nn.utils.clip_grad_norm(critic.parameters(), 1.)
        self.optim['critic'].step()

        model.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 1.)
        self.optim['model'].step()

        return torch.cat([final_loss.view(-1), p_losses.view(-1), e_losses.view(-1),
                          value_loss.view(-1),
                          torch.ones(1, device=final_loss.device) * policy_stats.mean_loss(0),
                          torch.ones(1, device=final_loss.device) * policy_stats.mean_loss(1),
                          old_p_losses.mean().view(-1) ],).view(1,-1).cpu().data.numpy()

    def validate(self, args, valid_critic=False):
        split = 'dev'
        self.model.eval()
        total_stats = RLStatistics()
        print('='*20, 'VALIDATION', '='*20)
        for scenario in self.scenarios[split][:200]:
            controller = self._get_controller(scenario, split=split)
            controller.sessions[self.training_agent].set_controller(controller)
            example = controller.simulate(args.max_turns, verbose=args.verbose)
            session = controller.sessions[self.training_agent]
            reward = self.get_reward(example, session)
            stats = RLStatistics(reward=reward, n_words=1)
            total_stats.update(stats)
        print('='*20, 'END VALIDATION', '='*20)
        self.model.train()
        return total_stats

    def save_best_checkpoint(self, checkpoint, opt, valid_stats):

        path = None
        if opt.model_type == 'reinforce' or opt.model_type == 'a2c':
            if self.best_valid_reward is None or valid_stats.mean_reward() > self.best_valid_reward:
                self.best_valid_reward = valid_stats.mean_reward()
                path = '{root}/{model}_best.pt'.format(
                            root=opt.model_path,
                            model=opt.model_filename)
        elif opt.model_type == 'critic':
            if self.best_valid_loss is None or valid_stats.mean_loss() < self.best_valid_loss:
                self.best_valid_loss = valid_stats.mean_loss()
                path = '{root}/{model}_best.pt'.format(
                            root=opt.model_path,
                            model=opt.model_filename)

        if path is not None:
            print('Save best checkpoint {path}'.format(path=path))
            torch.save(checkpoint, path)

    def checkpoint_path(self, episode, opt, stats):
        path=None
        if opt.model_type == 'reinforce' or opt.model_type == 'a2c':
            path = '{root}/{model}_reward{reward:.2f}_e{episode:d}.pt'.format(
                        root=opt.model_path,
                        model=opt.model_filename,
                        reward=stats.mean_reward(),
                        episode=episode)
        elif opt.model_type == 'critic':
            path = '{root}/{model}_loss{reward:.4f}_e{episode:d}.pt'.format(
                        root=opt.model_path,
                        model=opt.model_filename,
                        reward=stats.mean_loss(),
                        episode=episode)
        assert path is not None
        return path

    def update_opponent(self, type=None):
        if type is None:
            types = ['policy', 'critic']
        elif not isinstance(type, list):
            types = [type]
        else:
            types = type

        print('update opponent model for {}.'.format(types))
        if 'policy' in types:
            tmp_model_dict = self.agents[self.training_agent].env.model.state_dict()
            self.agents[self.training_agent^1].env.model.load_state_dict(tmp_model_dict)
        if 'critic' in types:
            tmp_model_dict = self.agents[self.training_agent].env.critic.state_dict()
            self.agents[self.training_agent^1].env.critic.load_state_dict(tmp_model_dict)

    def learn(self, args):
        rewards = [None]*2
        s_rewards = [None]*2

        critic_report_stats = RLStatistics()
        critic_stats = RLStatistics()
        last_time = time.time()

        tensorboard_every = 10

        history_train_losses = [[],[]]

        batch_size = 16

        for i in range(args.num_dialogues // batch_size):
            _batch_iters = []
            _rewards = []

            for j in range(batch_size):
                # Rollout
                scenario = self._get_scenario()
                controller = self._get_controller(scenario, split='train')
                controller.sessions[self.training_agent].set_controller(controller)
                example = controller.simulate(args.max_turns, verbose=args.verbose)

                for session_id, session in enumerate(controller.sessions):
                    # if args.only_run != True and session_id != self.training_agent:
                    #     continue
                    # Compute reward
                    reward = self.get_reward(example, session)
                    # Standardize the reward
                    all_rewards = self.all_rewards[session_id]
                    all_rewards.append(reward)
                    s_reward = (reward - np.mean(all_rewards)) / max(1e-4, np.std(all_rewards))

                    rewards[session_id] = reward
                    s_rewards[session_id] = s_reward

                for session_id, session in enumerate(controller.sessions):
                    # Only train one agent
                    if args.only_run != True and session_id != self.training_agent:
                        continue

                    batch_iter = session.iter_batches()
                    T = next(batch_iter)
                    _batch_iters.append(batch_iter)
                    _rewards.append(reward)


                # if train_policy:
                #     self.update(batch_iter, reward, self.model, discount=args.discount_factor)
                #
                # if train_critic:
                #     stats = self.update_critic(batch_iter, reward, self.critic, discount=args.discount_factor)
                #     critic_report_stats.update(stats)
                #     critic_stats.update(stats)
            loss = self.update_a2c(_batch_iters, _rewards, self.model, self.critic, discount=args.discount_factor)
            history_train_losses[self.training_agent].append(loss)

            # print('verbose: ', args.verbose)

            if args.verbose:
                # if train_policy or args.model_type == 'tom':
                from core.price_tracker import PriceScaler
                for session_id, session in enumerate(controller.sessions):
                    bottom, top = PriceScaler.get_price_range(session.kb)
                    print('Agent[{}: {}], bottom ${}, top ${}'.format(session_id, session.kb.role, bottom, top))


                strs = example.to_text()
                for str in strs:
                    print(str)
                print("reward: [0]{}\nreward: [1]{}".format(self.all_rewards[0][-1], self.all_rewards[1][-1]))
                    # print("Standard reward: [0]{} [1]{}".format(s_rewards[0], s_rewards[1]))

            # Save logs on tensorboard
            if (i + 1) % tensorboard_every == 0:
                for j in range(2):
                    self.writer.add_scalar('agent{}/reward'.format(j), np.mean(self.all_rewards[j][-tensorboard_every:]), i)
                    if len(history_train_losses[j]) >= tensorboard_every:
                        tmp = np.concatenate(history_train_losses[j][-tensorboard_every:], axis=0)
                        tmp = np.mean(tmp, axis=0)
                        self.writer.add_scalar('agent{}/total_loss'.format(j), tmp[0], i)
                        self.writer.add_scalar('agent{}/policy_loss'.format(j), tmp[1], i)
                        self.writer.add_scalar('agent{}/entropy_loss'.format(j), tmp[2], i)
                        self.writer.add_scalar('agent{}/value_loss'.format(j), tmp[3], i)
                        self.writer.add_scalar('agent{}/intent_loss'.format(j), tmp[4], i)
                        self.writer.add_scalar('agent{}/price_loss'.format(j), tmp[5], i)
                        self.writer.add_scalar('agent{}/logp_loss'.format(j), tmp[6], i)


            if ((i + 1) % args.report_every) == 0:
                import seaborn as sns
                import matplotlib.pyplot as plt
                if args.histogram:
                    sns.set_style('darkgrid')

                # if train_policy:
                for j in range(2):
                    print('agent={}'.format(j), end=' ')
                    print('step:', i, end=' ')
                    print('reward:', rewards[j], end=' ')
                    print('scaled reward:', s_rewards[j], end=' ')
                    print('mean reward:', np.mean(self.all_rewards[j]))
                    if args.histogram:
                        self.agents[j].env.dialogue_generator.get_policyHistogram()

                # if train_critic:
                #     critic_report_stats.output(i+1, 0, 0, last_time)
                #     critic_report_stats = RLStatistics()

                print('-'*10)
                if args.histogram:
                    plt.show()

                last_time = time.time()

            # Save model
            if (i > 0 and i % 100 == 0) and not args.only_run:
                # TODO: valid in dev set
                valid_stats = self.validate(args)
                self.drop_checkpoint(args, i, valid_stats, model_opt=self.agents[self.training_agent].env.model_args)
                self.update_opponent(['policy', 'critic'])

                # if train_policy:
                #     valid_stats = self.validate(args)
                #     self.drop_checkpoint(args, i, valid_stats, model_opt=self.agents[self.training_agent].env.model_args)
                #     self.update_opponent('policy')
                #
                # elif train_critic:
                #     # TODO: reverse!
                #     self.drop_checkpoint(args, i, critic_stats, model_opt=self.agents[self.training_agent].env.model_args)
                #     critic_stats = RLStatistics()
                # else:
                #     valid_stats = self.validate(args)
                #     print('valid result: ', valid_stats.str_loss())