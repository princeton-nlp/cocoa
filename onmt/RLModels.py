import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq

from .Models import RNNEncoder


class MeanEncoder(nn.Module):

    def __init__(self, input_size, ):
        super(MeanEncoder, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=1)
        return x


class RNNEncoder(nn.Module):

    def __init__(self, input_size, hidden_size=64, num_layers=1, type='LSTM'):
        super(RNNEncoder, self).__init__()
        # assert type in ['RNN', 'LSTM']
        self.rnn_type = type
        if type == 'RNN':
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError('Type:{} should be RNN or LSTM'.format(type))
        # self.output_layer = nn.Sequential(nn.Linear(hidden_size, input_size), nn.ReLU(input_size))

    def forward(self, x, h_state=None):
        r_out, h_state = self.rnn(x, h_state)
        # outs = []
        # for time in range(r_out.size(1)):
        #     outs.append(self.out(r_out[:, time, :]))
        return h_state[-1]

class UtteranceEncoder(nn.Module):
    def __init__(self, encoder, embedding, fix_emb=False):
        super(UtteranceEncoder, self).__init__()

        self.fix_emb = fix_emb
        self.emb = embedding

        self.encoder = encoder
        self.output_size = embedding.embedding_dim

    # utterance: (batch_size, )
    def forward(self, utterances):
        if self.fix_emb:
            with torch.no_grad():
                hidden_state = self.emb(utterances)
        else:
            # print('self.emb.embedding_dim', self.emb.embedding_dim)
            hidden_state = self.emb(utterances)
        hidden_state = self.encoder(hidden_state)
        return hidden_state


class StateEncoder(nn.Module):

    def __init__(self, intent_size, output_size=64, num_layer=1, hidden_size=64, state_length=2,
                 extra_size=0, ):
        super(StateEncoder, self).__init__()
        self.state_length = state_length
        self.extra_size = extra_size
        self.intent_size = intent_size
        self.output_size = output_size

        # intent | price | roles | number of history
        last_size = intent_size*state_length + state_length + extra_size

        hidden_layers = []
        for i in range(num_layer):
            hidden_layers += [nn.Linear(last_size, hidden_size), nn.ReLU(hidden_size)]
            last_size = hidden_size
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(last_size, output_size)

    def forward(self, e_intent, e_price, e_pmask, e_extra=None):
        # (batch, emb_dim * state_num)
        batch_size = e_intent.shape[0]
        old_et = e_intent
        e_intent = e_intent.reshape(-1, 1)
        e_intent = torch.zeros((e_intent.shape[0], self.intent_size), device=e_intent.device).scatter(1, e_intent, 1)
        e_intent = e_intent.reshape(batch_size, -1)
        # e_intent = self.embeddings(e_intent).view(-1, self.embeddings.embedding_dim*self.state_length)
        # print(e_price.shape, e_pmask.shape)
        e_price = e_price.mul(e_pmask).view(-1, self.state_length)
        # print(e_price.shape)

        # Used for info like current dialogue length.
        if e_extra is None:
            assert self.extra_size == 0
            e_extra = []
        else:
            # assert self.extra_size == e_extra.size(1)
            # print(e_intent.shape, e_price.shape, e_extra.shape)
            e_extra = [e_extra]

        if(np.any(np.isnan(e_intent.data.cpu().numpy()))):
            print('there is some nan')
            # print(e_intent.data, old_et)
            assert False

        e_output = torch.cat([e_intent, e_price]+e_extra, dim=1)


        # print(e_output.shape)
        e_output = self.hidden_layers(e_output)
        e_output = self.output_layer(e_output)

        return e_output


class StateUtteranceEncoder(nn.Module):
    def __init__(self, state_encoder, utterance_encoder, input_size=64*2, output_size=64, num_layer=1, hidden_size=64):
        super(StateUtteranceEncoder, self).__init__()
        self.state_encoder = state_encoder
        self.utterance_encoder = utterance_encoder

        hidden_layers = []
        last_size = input_size
        for i in range(num_layer):
            hidden_layers += [nn.Linear(last_size, hidden_size), nn.ReLU(hidden_size)]
            last_size = hidden_size
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(last_size, output_size)

    def forward(self, e_intent, e_price, e_pmask, e_extra=None, utterance=None):
        state_hidden_state = self.state_encoder(e_intent, e_price, e_pmask, e_extra)
        utterance_hidden_state = self.utterance_encoder(utterance)
        # print('utter:', state_hidden_state.shape, utterance_hidden_state.shape)
        hidden_state = torch.cat((state_hidden_state, utterance_hidden_state), dim=1)

        # print('utter:', hidden_state.shape)
        hidden_state = self.hidden_layers(hidden_state)
        hidden_state = self.output_layer(hidden_state)

        return hidden_state

class PolicyDecoder(nn.Module):

    max_price, min_price = 2, -0.5

    def __init__(self, encoder_size, intent_size, num_layer=2, hidden_size=128, use_utterance=False):
        super(PolicyDecoder, self).__init__()

        last_size = encoder_size
        hidden_layers = []
        for i in range(num_layer):
            hidden_layers += [nn.Linear(last_size, hidden_size), nn.ReLU(hidden_size)]
            last_size = hidden_size
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.intent_layer = nn.Linear(last_size, intent_size)
        self.price_layer = nn.Linear(last_size, 2)

        self.mu, self.logvar = None, None

    def reparameterize(self, mu, logvar):
        rdm = torch.randn_like(mu)
        if mu.device.type == 'gpu':
            rdm = rdm.cuda()
        eps = torch.autograd.Variable(rdm)
        z = mu + eps * torch.exp(logvar / 2)
        return z

    def forward(self, e_output):

        e_output = self.hidden_layers(e_output)

        policy = self.intent_layer(e_output)
        # policy = torch.softmax(policy.mul(intent_mask), dim=1)

        p_dis = self.price_layer(e_output)
        p_mean = p_dis[:,0]
        p_var  = p_dis[:,1]
        self.mu = p_mean
        # self.logvar = p_var
        self.logvar = p_var = torch.ones_like(self.mu) * math.log(0.1)
        # print('mean, var: ', p_mean.item(), torch.exp(p_var / 2).item(), self.mu, self.logvar)
        # if self.training:
        #     p = self.reparameterize(p_mean, p_var)
        # else:
        #     p = p_mean
        #
        # # TODO: test p_mean
        p = p_mean

        p_mean = p_mean.mul((p_mean < PolicyDecoder.max_price).float()) + \
                 torch.ones_like(p_mean, device=p_mean.device).mul((p_mean >= PolicyDecoder.max_price).float())
        p_mean = p_mean.mul((p_mean > PolicyDecoder.min_price).float()) + \
                 torch.zeros_like(p_mean, device=p_mean.device).mul((p_mean <= PolicyDecoder.min_price).float())
        # assert torch.max(p_mean) < 2
        # assert torch.max(p_mean) > -0.5
        # p = p_mean
        # p[p>1]=1
        # p[p<0]=0

        return policy, p_mean, p_var


class MultilayerPerceptron(nn.Module):

    def __init__(self, input_size, layer_size, layer_depth):
        super(MultilayerPerceptron, self).__init__()

        last_size = input_size
        hidden_layers = []
        for i in range(layer_depth):
            hidden_layers += [nn.Linear(last_size, layer_size), nn.ReLU(layer_size)]
            last_size = layer_size
        self.hidden_layers = nn.Sequential(*hidden_layers)

    def forward(self, input):
        return self.hidden_layers(input)


class HistoryEncoder(nn.Module):

    def __init__(self, diaact_size, last_lstm_size, extra_size, embeddings, output_size, hidden_size=64, hidden_depth=2):
        super(HistoryEncoder, self).__init__()

        self.fix_emb = False

        self.dia_lstm = torch.nn.LSTMCell(input_size=diaact_size, hidden_size=last_lstm_size)

        if embeddings is not None:
            uttr_emb_size = embeddings.embedding_dim
            self.uttr_lstm = torch.nn.LSTM(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)

            hidden_input = hidden_size + hidden_size + extra_size

        else:
            hidden_input = hidden_size + extra_size

        self.hidden_layer = MultilayerPerceptron(hidden_input, output_size, hidden_depth)

    def forward(self, diaact, uttr, extra, last_hidden):
        batch_size = diaact.shape[0]
        next_hidden = self.dia_lstm(diaact, last_hidden)
        dia_emb = next_hidden[0].reshape(batch_size, -1)

        if uttr is not None:
            uttr, lengths = uttr
            with torch.set_grad_enabled(not self.fix_emb):
                uttr = self.uttr_emb(uttr)
            uttr = torch.nn.utils.rnn.pack_padded_sequence(uttr, lengths, batch_first=True, enforce_sorted=False)
            _, output = self.uttr_lstm(uttr)

            # For LSTM case, output=(h_1, c_1)
            if isinstance(output, tuple):
                output = output[0]

            uttr_emb = output.reshape(batch_size, -1)

            hidden_input = torch.cat([dia_emb, uttr_emb, extra], dim=-1)
        else:

            hidden_input = torch.cat([dia_emb, extra], dim=-1)

        emb = self.hidden_layer(hidden_input)

        return emb, next_hidden


class CurrentEncoder(nn.Module):

    def __init__(self, input_size, embeddings, output_size, hidden_size=64, hidden_depth=2):
        super(CurrentEncoder, self).__init__()

        self.fix_emb = False

        if embeddings is not None:

            uttr_emb_size = embeddings.embedding_dim
            self.uttr_emb = embeddings
            self.uttr_lstm = torch.nn.LSTM(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)

            hidden_input = input_size + hidden_size

        else:
            hidden_input = input_size

        self.hidden_layer = MultilayerPerceptron(hidden_input, output_size, hidden_depth)

    def forward(self, uttr, extra):
        batch_size = extra.shape[0]

        if uttr is not None:
            uttr = uttr.copy()
            with torch.set_grad_enabled(not self.fix_emb):
                for i, u in enumerate(uttr):
                    if u.dtype != torch.int64:
                        print('uttr_emb:', uttr)
                    uttr[i] = self.uttr_emb(u).reshape(-1, self.uttr_emb.embedding_dim)
                # print(uttr[i].shape)
            uttr = torch.nn.utils.rnn.pack_sequence(uttr, enforce_sorted=False)
            # uttr = torch.nn.utils.rnn.pack_padded_sequence(uttr, lengths, batch_first=True, enforce_sorted=False)

            _, output = self.uttr_lstm(uttr)

            # For LSTM case, output=(h_1, c_1)
            if isinstance(output, tuple):
                output = output[0]

            uttr_emb = output.reshape(batch_size, -1)

            hidden_input = torch.cat([uttr_emb, extra], dim=-1)
        else:
            hidden_input = extra

        emb = self.hidden_layer(hidden_input)

        return emb


class SinglePolicy(nn.Module):

    def __init__(self, input_size, output_size, num_layer=1, hidden_size=128, ):
        super(SinglePolicy, self).__init__()

        self.hidden_layers = MultilayerPerceptron(input_size, hidden_size, num_layer)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, emb):
        hidden_state = self.hidden_layers(emb)
        output = self.output_layer(hidden_state)
        return output

        
# For RL Agent
# Intent + Price(action)
class MixedPolicy(nn.Module):

    def __init__(self, input_size, intent_size, price_size, hidden_size=64, hidden_depth=2, price_extra=0):
        super(MixedPolicy, self).__init__()
        self.common_net = MultilayerPerceptron(input_size, hidden_size, hidden_depth)
        self.intent_net = SinglePolicy(hidden_size, intent_size, num_layer=1, hidden_size=hidden_size)
        self.price_net = SinglePolicy(hidden_size + price_extra, price_size, num_layer=1, hidden_size=hidden_size//2)
        self.intent_size = intent_size
        self.price_size = price_size

    def forward(self, state_emb, price_extra=None):
        common_state = self.common_net(state_emb)

        intent_output = self.intent_net(common_state)

        price_input = [common_state]
        if price_extra:
            price_input.append(price_extra)
        price_input = torch.cat(price_input, dim=-1)
        price_output = self.price_net(price_input)

        return intent_output, price_output


class ValueDecoder(nn.Module):

    def __init__(self, encoder_size, num_layer=2, hidden_size=128):
        super(ValueDecoder, self).__init__()

        last_size = encoder_size
        hidden_layers = []
        for i in range(num_layer):
            hidden_layers += [nn.Linear(last_size, hidden_size), nn.ReLU(hidden_size)]
            last_size = hidden_size
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Linear(last_size, 1)

    def forward(self, e_output):
        e_output = self.hidden_layers(e_output)
        value = self.output_layer(e_output)

        return value


class CurrentModel(nn.Module):

    def __init__(self, encoder, decoder, fix_encoder=False):
        super(CurrentModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fix_encoder = fix_encoder

    def forward(self, *input):
        with torch.set_grad_enabled(not self.fix_encoder):
            emb = self.encoder(*input)
        output = self.decoder(emb)
        return output


class HistoryModel(nn.Module):

    def __init__(self, encoder, decoder, fix_encoder=False):
        super(HistoryModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fix_encoder = fix_encoder

    def forward(self, *input):
        with torch.set_grad_enabled(not self.fix_encoder):
            emb, next_hidden = self.encoder(*input)
        output = self.decoder(emb)
        return output, next_hidden

class PolicyModel(nn.Module):
    """
        Core trainable object in OpenNMT. Implements a trainable interface
        for a simple, generic encoder + decoder(output action) model.

        Args:
          encoder (:obj:`EncoderBase`): an encoder object
          decoder (:obj:`RNNDecoderBase`): a decoder object
          multi<gpu (bool): setup for multigpu support

          encoder + policy
        """

    def __init__(self, encoder, decoder, multigpu=False, fix_encoder=False):
        self.multigpu = multigpu
        super(PolicyModel, self).__init__()
        self.fix_encoder = fix_encoder
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, e_intent, e_price, e_pmask, e_extra=None, utterance=None):
        """
        """
        if utterance is not None:
            enc_final = self.encoder(e_intent, e_price, e_pmask, e_extra, utterance)
        else:
            enc_final = self.encoder(e_intent, e_price, e_pmask, e_extra)
        if self.fix_encoder:
            enc_final = enc_final.detach()
        policy, price, price_var = self.decoder(enc_final)
        return policy, price, price_var


class ValueModel(nn.Module):
    """
        encoder + value
    """

    def __init__(self, encoder, decoder, multigpu=False, fix_encoder=False):
        self.multigpu = multigpu
        super(ValueModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fix_encoder = fix_encoder
        # encoder.eval()

    def forward(self, e_intent, e_price, e_pmask, e_extra=None, utterance=None):
        if utterance is not None:
            enc_final = self.encoder(e_intent, e_price, e_pmask, e_extra, utterance)
        else:
            enc_final = self.encoder(e_intent, e_price, e_pmask, e_extra)
        if self.fix_encoder:
            enc_final = enc_final.detach()
        value = self.decoder(enc_final)
        return value

    # def eval(self):
    #     self.decoder.eval()
    #
    # def train(self, mode=True):
    #     self.docoder.train()