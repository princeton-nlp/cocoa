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
        self.price_actions = map(self.vocab.to_ind, ('init-price', 'counter-price', markers.OFFER))
        self.prices = set([id_ for w, id_ in self.vocab.word_to_ind.items() if is_entity(w)])
        self.price_list = list(self.prices)
        self.eos = self.vocab.to_ind(markers.EOS)
        # TODO: fix the hard coding
        actions = set([w for w in self.vocab.word_to_ind if not
                (is_entity(w) or w in category_markers or w in sequence_markers
                    or w in (vocab.UNK, '</sum>', '<slot>', '</slot>'))])
        self.actions = map(self.vocab.to_ind, actions)
        # for i,j in self.vocab.word_to_ind.items():
        #     print(i,j)
        print('special:{}'.format(self.vocab.ind_to_word))
        print('price_actions:{}'.format(list(self.price_actions)))
        print('price:{}'.format(self.prices))
        print('actions:{}'.format(list(self.actions)))
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

    def generate_batch(self, batch, gt_prefix=1, enc_state=None):
        # This is to ensure we can stop at EOS for stateful models
        assert batch.size == 1

        # (1) Run the encoder on the src.
        lengths = batch.lengths
        dec_states, enc_memory_bank, enc_output = self._run_encoder(batch, enc_state)
        memory_bank = self._run_attention_memory(batch, enc_memory_bank)

        # (1.1) Go over forced prefix.
        inp = batch.decoder_inputs[:gt_prefix]
        dec_out, dec_states, _ = self.model.decoder(
            inp, memory_bank, dec_states, memory_lengths=lengths)

        # (2) Sampling
        batch_size = batch.size
        preds = []
        for i in range(self.max_length):
            # Outputs to probs
            dec_out = dec_out.squeeze(0)  # (batch_size, rnn_size)
            out = self.model.generator.forward(dec_out).data  # Logprob (batch_size, vocab_size)
            # Sample with temperature
            scores = out.div(self.temperature)

            # Masking to ensure valid LF
            # NOTE: batch size must be 1. TODO: relax this restriction
            if i > 0:
                mask = torch.zeros(scores.size())
                if pred[0] in self.price_actions:
                    # Only price will be allowed
                    mask[:, self.price_list] = 1
                elif pred[0] in self.prices or pred[0] in self.actions:
                    # Must end
                    mask = torch.zeros(scores.size())
                    mask[:, self.eos] = 1
                else:
                    mask[:, :] = 1
                scores[mask == 0] = -100.

            scores.sub_(scores.max(1, keepdim=True)[0].expand(scores.size(0), scores.size(1)))
            #print('score: ', scores.exp())

            self.policy_history.append(torch.softmax(scores, dim=1).cpu().numpy())

            pred = torch.multinomial(scores.exp(), 1).squeeze(1)  # (batch_size,)
            #print('pred: ', pred)

            preds.append(pred)
            if pred[0] == self.eos:
                break
            # Forward step
            inp = Variable(pred.view(1, -1))  # (seq_len=1, batch_size)
            dec_out, dec_states, _ = self.model.decoder(
                inp, memory_bank, dec_states, memory_lengths=lengths)

        preds = torch.stack(preds).t()  # (batch_size, seq_len)
        # Insert one dimension (n_best) so that its structure is consistent
        # with beam search generator
        preds = preds.unsqueeze(1)
        # TODO: add actual scores
        ret = {"predictions": preds,
               "scores": [[0]] * batch_size,
               "attention": [None] * batch_size,
               "dec_states": dec_states,
               }

        ret["gold_score"] = [0] * batch_size
        ret["batch"] = batch
        ret["enc_output"] = enc_output[-1].detach()
        return ret

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
