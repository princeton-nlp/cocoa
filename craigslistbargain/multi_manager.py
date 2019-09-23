import argparse
import random
import json
import numpy as np

from onmt.Utils import use_gpu

from cocoa.core.util import read_json
from cocoa.core.schema import Schema
from cocoa.core.scenario_db import ScenarioDB
from cocoa.neural.loss import ReinforceLossCompute
import cocoa.options

from core.scenario import Scenario
from core.controller import Controller
from systems import get_system
from neural.rl_trainer import RLTrainer
from neural import build_optim
import options

from neural.a2c_trainer import RLStatistics

from tensorboardX import SummaryWriter

import os

try:
    import thread
except ImportError:
    import _thread as thread

import multiprocessing
import multiprocessing.connection
import pickle
import numpy as np

def execute_runner(runner, args, addr):
    runner(args, addr).run()

class MultiRunner:
    def __init__(self, args, addr):
        self.init_trainer(args)
        self.addr = self.get_real_addr(addr)
        self.conn = multiprocessing.connection.Client(self.addr)

    def init_trainer(self, args):
        if args.random_seed:
            random.seed(args.random_seed+os.getpid())
            np.random.seed(args.random_seed+os.getpid())

        schema = Schema(args.schema_path)
        scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario)
        valid_scenario_db = ScenarioDB.from_dict(schema, read_json(args.valid_scenarios_path), Scenario)

        # if len(args.agent_checkpoints) == 0
        # assert len(args.agent_checkpoints) <= len(args.agents)
        if len(args.agent_checkpoints) < len(args.agents):
            ckpt = [None] * 2
        else:
            ckpt = args.agent_checkpoints

        systems = [get_system(name, args, schema, False, ckpt[i]) for i, name in enumerate(args.agents)]

        rl_agent = 0
        system = systems[rl_agent]
        model = system.env.model
        loss = None
        # optim = build_optim(args, [model, system.env.critic], None)
        optim = {'model': build_optim(args, model, None),
                 'critic': build_optim(args, system.env.critic, None)}
        optim['critic']._set_rate(0.05)

        scenarios = {'train': scenario_db.scenarios_list, 'dev': valid_scenario_db.scenarios_list}
        from neural.a2c_trainer import RLTrainer as A2CTrainer
        trainer = A2CTrainer(systems, scenarios, loss, optim, rl_agent,
                             reward_func=args.reward, cuda=(len(args.gpuid) > 0), args=args)

        self.args = args
        self.trainer = trainer
        self.systems = systems

    def get_real_addr(self, addr):
        return addr

    def simulate(self, cmd):
        raise NotImplementedError

    def train(self, cmd):
        raise NotImplementedError

    def update_model(self, cmd):
        raise NotImplementedError

    def fetch_model(self, cmd):
        raise NotImplementedError

    def valid(self, cmd):
        raise NotImplementedError

    def save_model(self, cmd):
        raise NotImplementedError

    def run(self):
        while True:
            cmd = self.conn.recv()

            print('recv: ', cmd[0])
            if cmd[0] == 'quit':
                break
            elif cmd[0] == 'check':
                self.conn.send(['done'])
            elif cmd[0] == 'simulate':
                data = self.simulate(cmd[1:])
                self.conn.send(['done', pickle.dumps(data[:2])])
                # try:
                # except Exception, err:
                #     print(e)
                #     self.conn.send(['error'])
            elif cmd[0] == 'train':
                data = self.train(pickle.loads(cmd[1]))
                self.conn.send(['done', pickle.dumps(data)])
                # try:
                #     data = self.train(pickle.loads(cmd[1]))
                #     self.conn.send(['done', pickle.dumps(data)])
                # except:
                #     self.conn.send(['error'])
            elif cmd[0] == 'update_model':
                self.update_model((cmd[1],) + pickle.loads(cmd[2]))
                self.conn.send(['done'])
                # try:
                #     self.update_model(pickle.loads(cmd[1]))
                #     self.conn.send(['done'])
                # except:
                #     self.conn.send(['error'])

            elif cmd[0] == 'fetch_model':

                data = self.fetch_model(cmd[1:])
                self.conn.send(['done', pickle.dumps(data)])
                # try:
                #     data = self.fetch_model(cmd[1:])
                #     self.conn.send(['done', pickle.dumps(data)])
                # except:
                #     self.conn.send(['error'])
            elif cmd[0] == 'valid':
                data = self.valid(cmd[1])
                self.conn.send(['done', pickle.dumps(data)])

            elif cmd[0] == 'save_model':
                self.save_model(pickle.loads(cmd[1]))
                self.conn.send(['done'])

class MultiManager():
    def __init__(self, num_cpu, args, worker_class):
        self.local_workers = []
        self.worker_addr = []
        self.trainer_addr = []
        self.args = args

        for i in range(num_cpu):
            addr = ('localhost', 7000+i)
            self.worker_addr.append(addr)
            self.local_workers.append(multiprocessing.Process(target=execute_runner, args=(worker_class, args, addr)))
        # self.trainer = multiprocessing.Process(target=execute_runner, args=(trainer_class, args))
        self.trainer = self.local_workers[0]

        self.worker_listener = []
        for i, addr in enumerate(self.worker_addr):
            self.worker_listener.append(multiprocessing.connection.Listener(addr))
        self.worker_conn = []

        self.writer = SummaryWriter(logdir='logs/{}'.format(args.name))

    def run_local_workers(self):
        for w in self.local_workers:
            w.start()

    def update_worker_list(self):
        self.worker_conn = []
        for l in self.worker_listener:
            self.worker_conn.append(l.accept())
        return len(self.worker_conn)

    @staticmethod
    def allocate_tasks(num_worker, batch_size):
        ret = []
        while num_worker > 0:
            ret.append(batch_size // num_worker)
            batch_size -= ret[-1]
            num_worker -= 1

        print('allocate: {} workers, {} tasks, final list:{}'.format(num_worker, batch_size, ret))
        return ret

    def _draw_tensorboard(self, ii, losses, all_rewards):
        # print(all_rewards)
        for j in range(2):
            self.writer.add_scalar('agent{}/reward'.format(j), np.mean(all_rewards[j]), ii)
            if len(losses[j]) > 0:
                tmp = np.concatenate(losses[j], axis=0)
                tmp = np.mean(tmp, axis=0)
                # tmp = losses[j]
                self.writer.add_scalar('agent{}/total_loss'.format(j), tmp[0], ii)
                self.writer.add_scalar('agent{}/policy_loss'.format(j), tmp[1], ii)
                self.writer.add_scalar('agent{}/entropy_loss'.format(j), tmp[2], ii)
                self.writer.add_scalar('agent{}/value_loss'.format(j), tmp[3], ii)
                self.writer.add_scalar('agent{}/intent_loss'.format(j), tmp[4], ii)
                self.writer.add_scalar('agent{}/price_loss'.format(j), tmp[5], ii)
                self.writer.add_scalar('agent{}/logp_loss'.format(j), tmp[6], ii)

    def run(self):
        self.run_local_workers()
        args = self.args
        rewards = [None] * 2
        s_rewards = [None] * 2
        tensorboard_every = 1
        save_every = 50

        history_train_losses = [[], []]

        batch_size = 50

        pretrain_rounds = 3
        if args.only_run:
            batch_size = 1
            pretrain_rounds = 0

        save_every = max(1, save_every // batch_size)
        report_every = max(1, args.report_every // batch_size)

        max_epoch = args.num_dialogues // batch_size
        epoch = 0
        data_size = 0

        all_rewards = [[], []]

        num_worker = self.update_worker_list()
        for epoch in range(max_epoch):
            batches = []
            rewards = [[], []]
            print('Epoch {}/{} running...'.format(epoch,max_epoch))
            task_lists = self.allocate_tasks(num_worker, batch_size)

            # Use workers to get trajectories
            for i, w in enumerate(self.worker_conn):
                w.send(['simulate', epoch, batch_size, task_lists[i]])

            for i, w in enumerate(self.worker_conn):
                info = w.recv()
                if info[0] != 'done':
                    print('Error on {}: {}.'.format(i, info))
                data = pickle.loads(info[1])
                batches += data[0]
                rewards[0] += data[1][0]
                rewards[1] += data[1][1]

            # Train the model
            self.worker_conn[0].send(['train', pickle.dumps((epoch, batches, rewards[0]))])
            train_info = self.worker_conn[0].recv()
            if train_info[0] != 'done':
                print('Error on {}: {}.'.format(i, train_info))
                
            # Draw outputs on the tensorboard
            self._draw_tensorboard((epoch + 1) * batch_size, [[pickle.loads(train_info[1])],[]],
                                   rewards)

            # Get new model from trainer
            self.worker_conn[0].send(['fetch_model', 0])
            info = self.worker_conn[0].recv()
            data = info[1]

            # Save local checkpoint

            # Update all the worker
            for i, w in enumerate(self.worker_conn):
                if i == 0:
                    continue
                w.send(['update_model', 0, data])

            for i, w in enumerate(self.worker_conn):
                if i == 0:
                    continue
                w.recv()

            # Valid new model
            task_lists = self.allocate_tasks(num_worker, 200)
            now = 0
            for i, w in enumerate(self.worker_conn):
                w.send(['valid', (now, task_lists[i])])
                now += task_lists[i]

            valid_stats = RLStatistics()
            for i, w in enumerate(self.worker_conn):
                valid_info = w.recv()
                valid_stats.update(pickle.loads(valid_info[1]))

            # Save the model
            self.worker_conn[0].send(['save_model', pickle.dumps((epoch, valid_stats))])
            self.worker_conn[0].recv()

        self.quit_all_workers()
        self.join_local_workers()

    def quit_all_workers(self):
        for w in self.worker_conn:
            w.send(['quit'])

    def join_local_workers(self):
        for w in self.local_workers:
            w.join()
