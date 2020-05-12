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

    def __init__(self, input_size, layer_size, layer_depth, final_output=None):
        super(MultilayerPerceptron, self).__init__()

        last_size = input_size
        hidden_layers = []
        if layer_depth == 0:
            self.hidden_layers = nn.Identity()
        else:
            for i in range(layer_depth):
                if final_output is not None and i == layer_depth-1:
                    hidden_layers += [nn.Linear(last_size, final_output)]
                else:
                    hidden_layers += [nn.Linear(last_size, layer_size), nn.ReLU(layer_size)]
                last_size = layer_size
            self.hidden_layers = nn.Sequential(*hidden_layers)

    def forward(self, input):
        return self.hidden_layers(input)


class CurrentEncoder(nn.Module):
    # intent | price | roles | number of history
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


class HistoryIdentity(nn.Module):

    def __init__(self, diaact_size, last_lstm_size, extra_size, identity_dim,
                 hidden_size=64, hidden_depth=2, rnn_type='rnn'):
        super(HistoryIdentity, self).__init__()

        self.fix_emb = False
        self.identity_dim = identity_dim

        if rnn_type == 'lstm':
            self.dia_rnn = torch.nn.LSTMCell(input_size=diaact_size, hidden_size=last_lstm_size)
        else:
            self.dia_rnn = torch.nn.RNNCell(input_size=diaact_size, hidden_size=last_lstm_size)

        hidden_input = last_lstm_size + extra_size

        self.hidden_layer = MultilayerPerceptron(hidden_input, hidden_size, hidden_depth, final_output=identity_dim)

    def forward(self, diaact, extra, last_hidden):
        batch_size = diaact.shape[0]
        next_hidden = self.dia_rnn(diaact, last_hidden)
        if isinstance(next_hidden, tuple):
            # For LSTM
            dia_emb = next_hidden[0].reshape(batch_size, -1)
        else:
            # For RNN
            dia_emb = next_hidden.reshape(batch_size, -1)

        hidden_input = torch.cat([dia_emb, extra], dim=-1)

        emb = self.hidden_layer(hidden_input)

        return emb, next_hidden


class HistoryEncoder(nn.Module):
    """RNN Encoder

    """
    def __init__(self, diaact_size, extra_size, embeddings, output_size,
                 hidden_size=64, hidden_depth=2, rnn_type='rnn', fix_identity=True):
        super(HistoryEncoder, self).__init__()

        self.fix_emb = False
        last_lstm_size = hidden_size

        if rnn_type == 'lstm':
            self.dia_rnn = torch.nn.LSTMCell(input_size=diaact_size, hidden_size=last_lstm_size)
        else:
            self.dia_rnn = torch.nn.RNNCell(input_size=diaact_size, hidden_size=last_lstm_size)

        # hidden_input = last_lstm_size + extra_size
        #
        # self.hidden_layer = MultilayerPerceptron(hidden_input, hidden_size, hidden_depth, final_output=identity_dim)

        self.fix_emb = False
        self.fix_identity = fix_identity
        self.ban_identity = False

        self.uttr_emb = embeddings

        if embeddings is not None:
            uttr_emb_size = embeddings.embedding_dim
            if rnn_type == 'lstm':
                self.uttr_rnn = torch.nn.LSTM(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)
            else:
                self.uttr_rnn = torch.nn.RNN(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)

            hidden_input = hidden_size + last_lstm_size + extra_size
        else:
            hidden_input = last_lstm_size + extra_size

        self.hidden_layer = MultilayerPerceptron(hidden_input, output_size, hidden_depth)

    def forward(self, uttr, state, identity_state):
        # State Encoder
        diaact, extra, last_hidden = identity_state
        # identity, next_hidden = self.identity(*identity_state)
        batch_size = diaact.shape[0]
        next_hidden = self.dia_rnn(diaact, last_hidden)
        if isinstance(next_hidden, tuple):
            # For LSTM
            dia_emb = next_hidden[0].reshape(batch_size, -1)
        else:
            # For RNN
            dia_emb = next_hidden.reshape(batch_size, -1)

        # Uttr Encoder
        batch_size = state.shape[0]
        if uttr is not None:
            uttr = uttr.copy()
            with torch.set_grad_enabled(not self.fix_emb):
                for i, u in enumerate(uttr):
                    if u.dtype != torch.int64:
                        print('uttr_emb:', uttr)
                    # print('uttr_emb', next(self.uttr_emb.parameters()).device, u.device)
                    uttr[i] = self.uttr_emb(u).reshape(-1, self.uttr_emb.embedding_dim)
                # print(uttr[i].shape)
            uttr = torch.nn.utils.rnn.pack_sequence(uttr, enforce_sorted=False)
            # uttr = torch.nn.utils.rnn.pack_padded_sequence(uttr, lengths, batch_first=True, enforce_sorted=False)

            _, output = self.uttr_rnn(uttr)

            # For LSTM case, output=(h_1, c_1)
            if isinstance(output, tuple):
                output = output[0]

            uttr_emb = output.reshape(batch_size, -1)

            hidden_input = torch.cat([uttr_emb, extra, dia_emb], dim=-1)
        else:

            hidden_input = torch.cat([extra, dia_emb], dim=-1)

        emb = self.hidden_layer(hidden_input)

        return emb, next_hidden


class HistoryIDEncoder(nn.Module):

    def __init__(self, identity, extra_size, embeddings, output_size,
                 hidden_size=64, hidden_depth=2, rnn_type='rnn', fix_identity=True):
        super(HistoryIDEncoder, self).__init__()

        self.fix_emb = False
        self.fix_identity = fix_identity
        self.ban_identity = False

        self.identity = identity
        self.uttr_emb = embeddings
        identity_size = identity.identity_dim

        if embeddings is not None:
            uttr_emb_size = embeddings.embedding_dim
            if rnn_type == 'lstm':
                self.uttr_rnn = torch.nn.LSTM(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)
            else:
                self.uttr_rnn = torch.nn.RNN(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)

            hidden_input = hidden_size + extra_size + identity_size
        else:
            hidden_input = extra_size + identity_size

        self.hidden_layer = MultilayerPerceptron(hidden_input, output_size, hidden_depth)

    def forward(self, uttr, state, identity_state):
        identity, next_hidden = self.identity(*identity_state)
        batch_size = state.shape[0]
        if self.fix_identity:
            _identity = identity.detach()
        else:
            _identity = identity
        if self.ban_identity:
            _identity.fill_(0)
        _identity = torch.softmax(_identity, dim=1)
        if uttr is not None:
            uttr = uttr.copy()
            with torch.set_grad_enabled(not self.fix_emb):
                for i, u in enumerate(uttr):
                    if u.dtype != torch.int64:
                        print('uttr_emb:', uttr)
                    # print('uttr_emb', next(self.uttr_emb.parameters()).device, u.device)
                    uttr[i] = self.uttr_emb(u).reshape(-1, self.uttr_emb.embedding_dim)
                # print(uttr[i].shape)
            uttr = torch.nn.utils.rnn.pack_sequence(uttr, enforce_sorted=False)
            # uttr = torch.nn.utils.rnn.pack_padded_sequence(uttr, lengths, batch_first=True, enforce_sorted=False)

            _, output = self.uttr_rnn(uttr)

            # For LSTM case, output=(h_1, c_1)
            if isinstance(output, tuple):
                output = output[0]

            uttr_emb = output.reshape(batch_size, -1)

            hidden_input = torch.cat([uttr_emb, state, _identity], dim=-1)
        else:

            hidden_input = torch.cat([state, _identity], dim=-1)

        emb = self.hidden_layer(hidden_input)

        return emb, next_hidden, identity


class SinglePolicy(nn.Module):

    def __init__(self, input_size, output_size, hidden_depth=1, hidden_size=128, ):
        super(SinglePolicy, self).__init__()

        self.hidden_layers = MultilayerPerceptron(input_size, hidden_size, hidden_depth)
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
        self.intent_net = SinglePolicy(hidden_size, intent_size, hidden_depth=1, hidden_size=hidden_size)
        self.price_net = SinglePolicy(hidden_size + price_extra, price_size, hidden_depth=1, hidden_size=hidden_size//2)
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
            # return emb, next_hidden, (identity)
            e_output = self.encoder(*input)

        d_output = self.decoder(e_output[0])
        return (d_output,) + e_output[1:]

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