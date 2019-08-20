import torch
from torch.autograd import Variable

import onmt.io
from onmt.Utils import aeq, use_gpu

from cocoa.core.entity import is_entity
from cocoa.neural.generator import Generator, Sampler

from .symbols import markers, category_markers, sequence_markers
from .utterance import UtteranceBuilder


class LFSampler(Sampler):
    def __init__(self, model, vocab,
                 temperature=1, max_length=100, cuda=False):
        super(LFSampler, self).__init__(model, vocab, temperature=temperature, max_length=max_length, cuda=cuda)
        self.acc_or_rej = list(map(self.vocab.to_ind, (markers.ACCEPT, markers.REJECT)))
        self.offer = list(map(self.vocab.to_ind, (markers.OFFER, )))
        self.price_actions = list(map(self.vocab.to_ind, ('counter', 'propose', markers.OFFER)))

        # for i,j in self.vocab.word_to_ind.items():
        #     print(i,j)
        print('acc_rej:{}'.format(list(map(self.vocab.to_word, self.acc_or_rej))))
        print('offer:{}'.format(list(map(self.vocab.to_word, self.offer))))

        # Draw the distribution of prices
        # p_list = [self.vocab.to_word(i).canonical.value for i in self.price_list]
        # p_list = sorted(p_list)
        # print('plist:{}'.format(p_list))
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # sns.set()
        # sns.distplot(p_list, rug=True, bins=50)
        # plt.show()

        self.policy_history = []
        self.all_actions = self._get_all_actions()

    def generate_batch(self, batch, gt_prefix=1, enc_state=None, whole_policy=False, special_actions=None):
        # This is to ensure we can stop at EOS for stateful models
        assert batch.size == 1

        # Run the model
        policy, price = self.model(batch.encoder_intent, batch.encoder_price, batch.encoder_pmask)

        # Get embeddings of target
        # tgt_emb = self.model.encoder.embeddings(batch.target_intent)
        # tgt_emb = torch.cat([tgt_emb, batch.target_price], )

        # policy.sub_(policy.max(1, keepdim=True)[0].expand(policy.size(0), policy.size(1)))
        policy.sub_(policy.max(1, keepdim=True).expand(-1, policy.size(1)))
        mask = batch.policy_mask
        policy[mask == 0] = -100.
        p_exp = policy.exp()
        policy = p_exp / (torch.sum(p_exp, keepdim=True, dim=1))
        intent = torch.multinomial(policy, 1).squeeze(1)  # (batch_size,)

        # TODO: Not correct, I think.
        if intent in self.price_actions:
            price = None

        ret = {"intent": intent,
               "price": price,
               "policy": policy,
               }

        ret["batch"] = batch
        # ret["policies"] = policies
        # ret["probability"] = probs
        return ret

    def _get_all_actions(self):
        all_actions = []
        num_acions = len(self.vocab.word_to_ind)
        print(self.actions)
        print(list(self.actions))
        for i in list(self.actions):
            if not i in self.price_actions:
                print('a ', self.vocab.to_word(i))
                all_actions.append((i,))
        for i in self.price_actions:
            print('pa ', self.vocab.to_word(i))
            for j in self.prices:
                all_actions.append((i,j))
        return all_actions

    def get_policyHistogram(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import re

        allNum = len(self.policy_history)
        tmpData = np.mean(self.policy_history,axis=0)[0]
        r = re.compile(u'\d+[.]?\d*')
        x, w = [], []
        for i in range(len(tmpData)):
            tmp = self.vocab.ind_to_word[i]
            if not is_entity(tmp):
                continue
            name = tmp.canonical.value
            if abs(name) > 10.1: continue
            x.append(name)
            w.append(tmpData[i])

        w = w/np.sum(w)
        from scipy.stats import norm
        sns.distplot(x, bins=100, kde=False, hist_kws={'weights': w}, )
        #plt.show()



def get_generator(model, vocab, scorer, args, model_args):
    from onmt.Utils import use_gpu
    if args.sample:
        if model_args.model == 'lf2lf':
            generator = LFSampler(model, vocab, args.temperature,
                                max_length=args.max_length,
                                cuda=use_gpu(args))
        else:
            generator = Sampler(model, vocab, args.temperature,
                                max_length=args.max_length,
                                cuda=use_gpu(args))
    else:
        generator = Generator(model, vocab,
                              beam_size=args.beam_size,
                              n_best=args.n_best,
                              max_length=args.max_length,
                              global_scorer=scorer,
                              cuda=use_gpu(args),
                              min_length=args.min_length)
    return generator
