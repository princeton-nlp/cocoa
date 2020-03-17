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
import pickle as pkl

from neural.batcher_rl import RLBatch, RawBatch, ToMBatch

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
        # print('training_agent', training_agent)

        self.critic = agents[training_agent].env.critic
        self.tom = agents[training_agent].env.tom_model
        self.model_type = args.model_type
        self.use_utterance = False
        self.tom_identity_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def _run_batch_a2c(self, batch):
        value = self._run_batch_critic(batch)
        policy, price = self._run_batch(batch)
        # print('max price', torch.max(price))
        return value, policy, price

    def _run_batch_tom_identity(self, batch, hidden_state):
        predictions, next_hidden = self.tom(batch.state, None, batch.extra, hidden_state)
        return predictions, next_hidden

    def _tom_gradient_accumulation(self, batch_iter, strategy, model, ):
        model.train()

        h = None
        preds = []
        losses = []
        accus = []
        strategies = []
        for i, batch in enumerate(batch_iter):
            tom_batch = ToMBatch.from_raw(batch, strategy)
            if h is not None:
                if isinstance(h, tuple):
                    h = (h[0][:batch.size, :], h[1][:batch.size, :])
                elif isinstance(h, torch.Tensor):
                    h = h[:batch.size, :]
            pred, h = self._run_batch_tom_identity(tom_batch, hidden_state=h)
            s = torch.tensor(strategy[:batch.size], dtype=torch.int64, device=pred.device)
            loss = self.tom_identity_loss(pred, s)
            accu = torch.gather(torch.softmax(pred, dim=1), 1, s.reshape(-1, 1))
            losses.append(loss.reshape(-1))
            accus.append(accu.reshape(-1))
            strategies.append(s.detach())
            preds.append(pred.reshape(1, -1).detach())

        # preds = torch.cat(preds, dim=0)
        # strategy = torch.tensor([strategy]*preds.shape[0], dtype=torch.int64, device=preds.device)
        # (-1,), (-1, 1) -> (-1,) *2
        # print('loss & accu:', loss, accu)
        return losses, accus, (preds, strategies)

    def _sort_merge_batch(self, batch_iters, batch_size, device=None):
        sorted_id = [i for i in range(len(batch_iters))]
        sorted_id.sort(key=lambda i: len(batch_iters[i]), reverse=True)
        batch_iters = sorted(batch_iters, reverse=True, key=lambda l: len(l))
        batch_length = [len(b) for b in batch_iters]

        if device is None:
            if self.cuda:
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')

        def merge_batch(one_batch):
            batches = [[] for i in range(len(one_batch[0]))]
            for bi in one_batch:
                for i, b in enumerate(bi):
                    batches[i].append(b)
            for i, b in enumerate(batches):
                # print('merge batch:', i, len(b))
                batches[i] = RawBatch.merge(b)
                batches[i].to(device)
            return batches

        # Split by size
        right = 0
        bs, ids, bl = [], [], []
        while True:
            left = right
            right = min(right+batch_size, len(batch_iters))
            # print('merge: ', left, right)
            bs.append(merge_batch(batch_iters[left: right]))
            ids.append(sorted_id[left: right])
            bl.append(batch_length[left: right])
            if right >= len(batch_iters):
                break

        return bs, ids, bl

    def update_tom(self, args, batch_iters, strategy, model, update_model=True, dump_name=None):
        # print('optim', type(self.optim['tom']))
        # cur_t = time.time()
        # batch_iters, sorted_id, batch_length = self._sort_merge_batch(batch_iters, 1024)
        batch_iters, sorted_id, batch_length = batch_iters
        # print('merge batch: {}s.'.format(time.time() - cur_t))
        # cur_t = time.time()

        model.zero_grad()
        loss = []
        step_loss = [[] for i in range(20)]
        step_accu = [[] for i in range(20)]
        output_data = []
        for i, b in enumerate(batch_iters):
            stra = [strategy[j] for j in sorted_id[i]]
            l, a, tmp = self._tom_gradient_accumulation(b, stra, model)
            output_data.append(tmp)
            # weight = torch.ones_like(l, device=l.device)
            # for j in range(5):
            #     if j < l.shape[0]:
            #         weight[j] = 5-j
            # l = l.mul(weight)
            # for j, ll in enumerate(l):
            #     if j == len(l)-1:
            #         loss.append(ll)
            #     else:
            #         if l[j+1].shape[0] > ll.shape[0]:
            #             loss.append(ll[l[j+1].shape[0]:])
            loss.append(torch.cat(l, dim=0))
            for j in range(len(l)):
                step_loss[j].append(l[j])
                step_accu[j].append(a[j])

        # print('calculate loss: {}s.'.format(time.time() - cur_t))
        # cur_t = time.time()

        loss = torch.cat(loss, dim=0).mean()

        step_num = [np.sum([dd.shape[0] for dd in d]) for d in step_loss]
        step_loss = [torch.cat(d, dim=0).mean().item() if len(d)>0 else None for d in step_loss]
        step_accu = [torch.cat(d, dim=0).mean().item() if len(d)>0 else None for d in step_accu]

        # print('returen infos: {}s.'.format(time.time() - cur_t))
        # cur_t = time.time()

        # dump output data
        if dump_name is not None:
            with open(dump_name, 'wb') as f:
                pkl.dump(output_data, f)

        # update
        if update_model:
            loss.backward()
            self.optim['tom'].step()
        # print('udpate model: {}s.'.format(time.time() - cur_t))
        # cur_t = time.time()


        return loss.item(), (step_loss, step_accu, step_num)

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
        for_value = False

        for i, batch in enumerate(batch_iter):
            # print("batch: \nencoder{}\ndecoder{}\ntitle{}\ndesc{}".format(batch.encoder_inputs.shape, batch.decoder_inputs.shape, batch.title_inputs.shape, batch.desc_inputs.shape))
            # batch.mask_last_price()
            rlbatch = RLBatch.from_raw(batch, None, None)
            value, policy, price = self._run_batch_a2c(rlbatch)
            # print('train_policy is:', policy)
            if not for_value:
                policy_loss, pl_stats = self._compute_loss(rlbatch, policy=policy, price=price, loss=self.train_loss)
                # print('policy_loss is:', policy_loss)
                policy_stats.update(pl_stats)

            entropy_loss, _ = self._compute_loss(rlbatch, policy=policy, price=price, loss=self.entropy_loss)

            # penalty = ((price-1)**2).mul((price>2).float()) + ((price-0)**2).mul((price<0.5).float())
            # penalty = ((price > 2).float()).mul((price - 1) ** 2) + ((price < 0.5).float()).mul((price - 0) ** 2)
            penalty = ((price > 2).float()).mul(0.1) + ((price < 0.5).float()).mul(0.1)
            penalty = torch.zeros_like(price, device=price.device)

            if not for_value:
                penalties.append(penalty.view(-1))
                p_losses.append(policy_loss.view(-1))
                e_losses.append(entropy_loss.view(-1))
            values.append(value.view(-1))

        # print('allnll ', nll)
        rewards = [0] * len(values)
        rewards[-1] = torch.ones(1) * reward

        old_rewards = [0] * len(values)
        old_rewards[-1] = torch.ones(1) * reward
        for i in range(len(rewards) - 2, -1, -1):
            rewards[i] = torch.ones(1) * rewards[i]
            old_rewards[i] = old_rewards[i + 1] * discount

        for i in range(len(rewards) - 2, -1, -1):
            rewards[i] += (values[i + 1].cpu().item()) * discount

        # print(old_rewards)

        new_rewards = torch.cat(rewards)
        new_values = torch.cat(values)
        if for_value:
            old_rewards = torch.cat(old_rewards[:-1])
            rewards = torch.cat(rewards[:-1])
            values = torch.cat(values[:-1])  # (total_seq_len, batch_size)
        else:
            old_rewards = torch.cat(old_rewards)
            rewards = torch.cat(rewards)
            values = torch.cat(values)  # (total_seq_len, batch_size)

        if self.cuda:
            new_rewards = new_rewards.cuda()
            new_values = new_values.cuda()
            old_rewards = old_rewards.cuda()
            rewards = rewards.cuda()
            values = values.cuda()


        value_loss, vl_stats = self._compute_loss(None, value=new_values, oracle=new_rewards, loss=self.critic_loss)
        # print('values', values, p_losses)
        old_p_losses = torch.cat(p_losses).view(rewards.shape)
        # p_losses = old_p_losses.mul(old_rewards).mean()
        # print('shapes', old_p_losses, old_rewards)
        # p_losses = old_p_losses.mul(old_rewards).mean()
        if self.model_type == 'reinforce':
            p_losses = old_p_losses.mul(old_rewards)
        else:
            p_losses = old_p_losses.mul(rewards - values.detach())
        e_losses = torch.cat(e_losses)
        regular = torch.cat(penalties)
        return p_losses, e_losses, value_loss, regular, (old_p_losses, policy_stats)

    def update_a2c(self, args, batch_iters, rewards, model, critic, discount=0.95, fix_policy=False, fix_value=False):
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

        # print('all loss', final_loss, p_losses, e_losses, value_loss)
        assert not torch.isnan(final_loss)
        # final_loss.backward()
        # model_loss.backward()
        # critic_loss.backward()
        # nn.utils.clip_grad_norm(critic.parameters(), 1.)
        # nn.utils.clip_grad_norm(model.parameters(), 1.)
        # self.optim.step()

        # if not self.model_type == "reinforce":
        if not args.only_run:
            if not fix_value:
                critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.)
                self.optim['critic'].step()

            if not fix_policy:
                model.zero_grad()
                model_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                self.optim['model'].step()

        return torch.cat([final_loss.view(-1), p_losses.view(-1), e_losses.view(-1),
                          value_loss.view(-1),
                          torch.ones(1, device=final_loss.device) * policy_stats.mean_loss(0),
                          torch.ones(1, device=final_loss.device) * policy_stats.mean_loss(1),
                          old_p_losses.mean().view(-1) ],).view(1,-1).cpu().data.numpy()

    def validate(self, args, valid_size, valid_critic=False, start=0, split='dev', exchange=None):
        rate = 0.5
        if exchange is not None:
            if exchange:
                rate = 1
            else:
                rate = 0
        self.model.eval()
        self.critic.eval()
        total_stats = RLStatistics()
        oppo_total_stats = RLStatistics()
        valid_size = min(valid_size, 200)
        # print('='*20, 'VALIDATION', '='*20)
        examples = []
        verbose_str = []
        for sid, scenario in enumerate(self.scenarios[split][start:start+valid_size]):
            controller = self._get_controller(scenario, split=split, rate=rate)
            controller.sessions[0].set_controller(controller)
            controller.sessions[1].set_controller(controller)
            example = controller.simulate(args.max_turns, verbose=args.verbose)
            session = controller.sessions[self.training_agent]
            reward = self.get_reward(example, session)
            rewards = [self.get_reward(example, controller.sessions[i]) for i in range(2)]
            stats = RLStatistics(reward=rewards[0], n_words=1)
            oppo_stats = RLStatistics(reward=rewards[1], n_words=1)
            total_stats.update(stats)
            oppo_total_stats.update(oppo_stats)
            examples.append(example)
            verbose_str.append(self.example_to_str(example, controller, rewards, sid+start))
        # print('='*20, 'END VALIDATION', '='*20)
        self.model.train()
        self.critic.train()
        return [total_stats, oppo_total_stats], examples, verbose_str

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
        if opt.model_type == 'reinforce' or opt.model_type == 'a2c' or opt.model_type == 'tom':
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

    def get_temperature(self, epoch, batch_size, args):
        if args.only_run or args.warmup_epochs == 0:
            return 1
        half = args.num_dialogues // batch_size / 2
        t_s, t_e = 0.3, 1
        i_s, i_e = 0, half
        return min(t_e, t_s + (t_e - t_s) * 1. * epoch / args.warmup_epochs)
        # return min(1., 1.*epoch/half)
    
    def example_to_text(self, exmaple):
        ret = []
        for i, e in enumerate(exmaple.events):
            if "real_uttr" in e.metadata.keys():
                ret.append("[{}: {}]\t{}\t{}\t\"{}\"".format(e.time, e.agent, e.action, e.data, e.metadata["real_uttr"]))
            else:
                ret.append("[{}: {}]\t{}\t{}".format(e.time, e.agent, e.action, e.data))
                ret.append("        \t{}\t{}".format(e.metadata.get('intent'), e.metadata.get('price')))
        return ret 
        

    def example_to_str(self, example, controller, rewards, sid=None):
        verbose_str = []
        from core.price_tracker import PriceScaler
        if sid is not None:
            verbose_str.append('[Scenario id: {}]'.format(sid))
        for session_id, session in enumerate(controller.sessions):
            bottom, top = PriceScaler.get_price_range(session.kb)
            s = 'Agent[{}: {}], bottom ${}, top ${}'.format(session_id, session.kb.role, bottom, top)
            verbose_str.append(s)
        verbose_str.append("They are negotiating for "+session.kb.facts['item']['Category'])

        strs = self.example_to_text(example)
        for str in strs:
            verbose_str.append(str)
        s = "reward: [0]{}\nreward: [1]{}".format(rewards[0], rewards[1])
        verbose_str.append(s)
        return verbose_str

    def sample_data(self, i, sample_size, args, real_batch=None, batch_size=128):
        if real_batch is None:
            real_batch = sample_size
        rewards = [0]*2
        s_rewards = [0]*2
        _batch_iters = [[], []]
        _rewards = [[], []]
        examples = []
        verbose_strs = []
        strategies = [[], []]

        dialogue_batch = [[], []]
        for j in range(real_batch):
            # Rollout
            scenario, sid = self._get_scenario()
            controller = self._get_controller(scenario, split='train')
            controller.sessions[0].set_controller(controller)
            controller.sessions[1].set_controller(controller)
            example = controller.simulate(args.max_turns, verbose=args.verbose, temperature=self.get_temperature(i, sample_size, args))

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
                _rewards[session_id].append(reward)
                strategies[session_id].append(session.price_strategy_label)

            for session_id, session in enumerate(controller.sessions):
                # dialogue_batch[session_id].append(session.dialogue)
                # if len(dialogue_batch[session_id]) == batch_size or j == real_batch-1:
                batch_iter = session.iter_batches()
                T = next(batch_iter)
                _batch_iters[session_id].append(list(batch_iter))


                # if train_policy or args.model_type == 'tom':

            examples.append(example)
            verbose_str = self.example_to_str(example, controller, rewards, sid)

            if args.verbose:
                for s in verbose_str:
                    print(s)
            verbose_strs.append(verbose_str)

        return _batch_iters, (_rewards, strategies), examples, verbose_strs

    def learn(self, args):
        rewards = [None]*2
        s_rewards = [None]*2

        critic_report_stats = RLStatistics()
        critic_stats = RLStatistics()
        last_time = time.time()

        tensorboard_every = 1
        save_every = 100

        history_train_losses = [[],[]]

        batch_size = 100

        pretrain_rounds = 3
        if args.only_run:
            batch_size = 1
            pretrain_rounds = 0

        save_every = max(1, save_every // batch_size)
        report_every = max(1, args.report_every // batch_size)

        for i in range(args.num_dialogues // batch_size):
            _batch_iters, _rewards, example, train_ex_str = self.sample_data(i, batch_size, args)
            # print('reward is:', _rewards)
            # print(np.mean(_rewards[0]), np.mean(_rewards[1]))
            # print(np.mean(self.all_rewards[0][-tensorboard_every*batch_size:]), np.mean(self.all_rewards[1][-tensorboard_every*batch_size:]))

            path_txt = '{root}/{model}_example{epoch}.txt'.format(
                root=args.model_path,
                model=args.name,
                epoch=i)
            with open(path_txt, 'w') as f:
                for ex in train_ex_str:
                    f.write('-' * 7 + '\n')
                    for s in ex:
                        f.write(s + '\n')

                # if train_policy:
                #     self.update(batch_iter, reward, self.model, discount=args.discount_factor)
                #
                # if train_critic:
                #     stats = self.update_critic(batch_iter, reward, self.critic, discount=args.discount_factor)
                #     critic_report_stats.update(stats)
                #     critic_stats.update(stats)
            k = -1
            for k in range(pretrain_rounds):
                loss = self.update_a2c(args, _batch_iters, _rewards[self.training_agent], self.model, self.critic,
                                       discount=args.discount_factor, fix_policy=True)
                # if (k+1)%5 == 0:
                #     _batch_iters, _rewards, example, _ = self.sample_data(i, batch_size, args)
                # if loss[0,3].item() < 0.2:
                #     break
            if k >=0:
                print('Pretrained value function for {} rounds, and the final loss is {}.'.format(k+1, loss[0,3].item()))
            # if loss[0, 3].item() >= 0.3:
            #     print('Try to initialize critic parameters.')
            #     for p in self.critic.parameters():
            #         p.data.uniform_(-args.param_init, args.param_init)
            #     for k in range(20):
            #         loss = self.update_a2c(args, _batch_iters, _rewards, self.model, self.critic,
            #                                discount=args.discount_factor, fix_policy=True)
            #         if (k + 1) % 5 == 0:
            #             _batch_iters, _rewards, controller, example = self.sample_data(i, batch_size, args)
            #         if loss[0, 3].item() < 0.2:
            #             break
            #     print('Pretrained value function for {} rounds, and the final loss is {}.'.format(k + 1,
            #                                                                                       loss[0, 3].item()))
            loss = self.update_a2c(args, _batch_iters, _rewards[self.training_agent], self.model, self.critic,
                                   discount=args.discount_factor)
            for k in range(pretrain_rounds):
                loss = self.update_a2c(args, _batch_iters, _rewards[self.training_agent], self.model, self.critic,
                                       discount=args.discount_factor, fix_policy=True)
            history_train_losses[self.training_agent].append(loss)

            # print('verbose: ', args.verbose)

                    # print("Standard reward: [0]{} [1]{}".format(s_rewards[0], s_rewards[1]))

            # Save logs on tensorboard
            if (i + 1) % tensorboard_every == 0:
                ii = (i+1)*batch_size
                for j in range(2):
                    self.writer.add_scalar('agent{}/reward'.format(j), np.mean(self.all_rewards[j][-tensorboard_every*batch_size:]), ii)
                    if len(history_train_losses[j]) >= tensorboard_every*batch_size:
                        tmp = np.concatenate(history_train_losses[j][-tensorboard_every*batch_size:], axis=0)
                        tmp = np.mean(tmp, axis=0)
                        self.writer.add_scalar('agent{}/total_loss'.format(j), tmp[0], ii)
                        self.writer.add_scalar('agent{}/policy_loss'.format(j), tmp[1], ii)
                        self.writer.add_scalar('agent{}/entropy_loss'.format(j), tmp[2], ii)
                        self.writer.add_scalar('agent{}/value_loss'.format(j), tmp[3], ii)
                        self.writer.add_scalar('agent{}/intent_loss'.format(j), tmp[4], ii)
                        self.writer.add_scalar('agent{}/price_loss'.format(j), tmp[5], ii)
                        self.writer.add_scalar('agent{}/logp_loss'.format(j), tmp[6], ii)


            if ((i + 1) % report_every) == 0:
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
                    print('mean reward:', np.mean(self.all_rewards[j][-args.report_every:]))
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
            if (i+1) % save_every == 0:
                # TODO: valid in dev set
                valid_stats, _, _ = self.validate(args, 50 if args.only_run else 200)
                valid_stats = valid_stats[0]
                if not args.only_run:
                    self.drop_checkpoint(args, i+1, valid_stats, model_opt=self.agents[self.training_agent].env.model_args)
                    if args.update_oppo:
                        print('update oppo!')
                        self.update_opponent(['policy', 'critic'])
                else:
                    print('valid ', valid_stats.str_loss())

                # if train_policy:
                #     valid_stats, _ = self.validate(args)
                #     self.drop_checkpoint(args, i, valid_stats, model_opt=self.agents[self.training_agent].env.model_args)
                #     self.update_opponent('policy')
                #
                # elif train_critic:
                #     # TODO: reverse!
                #     self.drop_checkpoint(args, i, critic_stats, model_opt=self.agents[self.training_agent].env.model_args)
                #     critic_stats = RLStatistics()
                # else:
                #     valid_stats, _ = self.validate(args)
                #     print('valid result: ', valid_stats.str_loss())