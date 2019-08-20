from cocoa.neural.trainer import Trainer as BaseTrainer, Statistics
import onmt
import torch
from torch import nn
from onmt.Utils import use_gpu
from cocoa.io.utils import create_path

class SimpleLoss(nn.Module):
    def __init__(self, inp_with_sfmx=False):
        super(SimpleLoss, self).__init__()
        if inp_with_sfmx:
            self.criterion_intent = nn.NLLLoss()
        else:
            self.criterion_intent = nn.CrossEntropyLoss()
        self.criterion_price = nn.MSELoss()
        self.use_nll = inp_with_sfmx

    def forward(self, enc_policy, enc_price, tgt_policy, tgt_price):
        loss0 = self.criterion_intent(enc_policy, tgt_policy)
        loss1 = self.criterion_price(enc_price, enc_price)
        loss = loss0 + loss1
        loss_data = loss.data.clone()
        stats = self._stats(loss_data, enc_policy.shape[0])
        return loss, stats

    def _stats(self, loss, num):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        return onmt.Statistics(loss.item(), num)


class SLTrainer(BaseTrainer):

    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 grad_accum_count=1):
        # Basic attributes.
        self.model = model
        self.train_loss = \
        self.valid_loss = SimpleLoss(inp_with_sfmx=False)
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.best_valid_loss = None
        self.cuda=False

        # Set model in training mode.
        self.model.train()

    ''' Class that controls the training process which inherits from Cocoa '''

    def _compute_loss(self, batch, policy, price, loss):
        # Get one-hot vectors of target
        class_num = len(batch.vocab)
        batch_size = batch.size
        # print('class_num {}\tbatch_size{}'.format(class_num, batch_size))
        # target_intent = torch.zeros(batch_size, class_num)
        # if batch.target_intent.device.type == 'cuda':
        #     target_intent = target_intent.cuda()
        # target_intent = target_intent.scatter_(1, batch.target_intent, 1)
        target_intent = batch.target_intent
        price = price.unsqueeze(1).mul(batch.target_pmask)
        # print('(policy, price, target_intent, batch.target_price)', (policy, price, target_intent, batch.target_price))
        return loss(policy, price, target_intent.squeeze(), batch.target_price)

    def _run_batch(self, batch, dec_state=None, enc_state=None):

        e_intent, e_price, e_pmask = batch.encoder_intent, batch.encoder_price, batch.encoder_pmask
        # print('e_intent {}\ne_price{}\ne_pmask{}'.format(e_intent, e_price, e_pmask))

        policy, price = self.model(e_intent, e_price, e_pmask)
        return policy, price

    def learn(self, opt, data, report_func):
        """Train model.
        Args:
            opt(namespace)
            model(Model)
            data(DataGenerator)
        """
        print('\nStart training...')
        print(' * number of epochs: %d' % opt.epochs)
        print(' * batch size: %d' % opt.batch_size)

        for epoch in range(opt.epochs):
            print('')

            # 1. Train for one epoch on the training set.
            train_iter = data.generator('train', cuda=use_gpu(opt))
            train_stats = self.train_epoch(train_iter, opt, epoch, report_func)
            print('Train loss: %g' % (train_stats.loss / train_stats.n_words))

            # 2. Validate on the validation set.
            valid_iter = data.generator('dev', cuda=use_gpu(opt))
            valid_stats = self.validate(valid_iter)
            print('Validation loss: %g' % (valid_stats.loss / train_stats.n_words))

            # 3. Log to remote server.
            # if opt.exp_host:
            #    train_stats.log("train", experiment, optim.lr)
            #    valid_stats.log("valid", experiment, optim.lr)
            # if opt.tensorboard:
            #    train_stats.log_tensorboard("train", writer, optim.lr, epoch)
            #    train_stats.log_tensorboard("valid", writer, optim.lr, epoch)

            # 4. Update the learning rate
            self.epoch_step(valid_stats.ppl(), epoch)

            # 5. Drop a checkpoint if needed.
            if epoch >= opt.start_checkpoint_at:
                self.drop_checkpoint(opt, epoch, valid_stats)

    def train_epoch(self, train_iter, opt, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        # Set model back to training mode.
        self.model.train()

        total_stats = Statistics()
        report_stats = Statistics()
        true_batchs = []
        accum = 0
        normalization = 0
        num_batches = next(train_iter)
        self.cuda = use_gpu(opt)

        for batch_idx, batch in enumerate(train_iter):
            true_batchs.append(batch)
            accum += 1

            if accum == self.grad_accum_count:
                self._gradient_accumulation(true_batchs, total_stats, report_stats)
                true_batchs = []
                accum = 0

            if report_func is not None:
                report_stats = report_func(opt, epoch, batch_idx, num_batches,
                                           total_stats.start_time, report_stats)

        # Accumulate gradients one last time if there are any leftover batches
        # Should not run for us since we plan to accumulate gradients at every
        # batch, so true_batches should always equal candidate batches
        if len(true_batchs) > 0:
            self._gradient_accumulation(true_batchs, total_stats, report_stats)
            true_batchs = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.

        self.model.eval()

        stats = Statistics()

        num_val_batches = next(valid_iter)
        for batch in valid_iter:
            if batch is None:
                continue
            policy, price = self._run_batch(batch)
            loss, batch_stats = self._compute_loss(batch, policy, price, self.train_loss)
            stats.update(batch_stats)

        # Set model back to training mode
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, valid_stats, model_opt=None):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        checkpoint = {
            'model': model_state_dict,
            'opt': opt if not model_opt else model_opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        path = self.checkpoint_path(epoch, opt, valid_stats)
        create_path(path)
        if not opt.best_only:
            print('Save checkpoint {path}'.format(path=path))
            torch.save(checkpoint, path)

        self.save_best_checkpoint(checkpoint, opt, valid_stats)

    def save_best_checkpoint(self, checkpoint, opt, valid_stats):
        if self.best_valid_loss is None or valid_stats.mean_loss() < self.best_valid_loss:
            self.best_valid_loss = valid_stats.mean_loss()
            path = '{root}/{model}_best.pt'.format(
                root=opt.model_path,
                model=opt.model_filename)

            print('Save best checkpoint {path}'.format(path=path))
            torch.save(checkpoint, path)

    def checkpoint_path(self, epoch, opt, stats):
        path = '{root}/{model}_loss{loss:.2f}_e{epoch:d}.pt'.format(
            root=opt.model_path,
            model=opt.model_filename,
            loss=stats.mean_loss(),
            epoch=epoch)
        return path

    def _gradient_accumulation(self, true_batchs, total_stats, report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        dec_state = None
        for batch in true_batchs:
            if batch is None:
                continue

            self.model.zero_grad()
            policy, price = self._run_batch(batch)

            loss, batch_stats = self._compute_loss(batch, policy, price, self.train_loss)
            loss.backward()
            self.optim.step()

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)
