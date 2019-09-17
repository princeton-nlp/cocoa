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


class StateEncoder(nn.Module):

    def __init__(self, embedding, output_size=64, num_layer=1, hidden_size=64, state_length=2, extra_size=0, fix_emb=False):
        super(StateEncoder, self).__init__()
        self.embeddings = embedding
        self.state_length = state_length
        self.extra_size = extra_size
        self.fix_emb = fix_emb

        last_size = embedding.embedding_dim*state_length + state_length + extra_size
        hidden_layers = []
        for i in range(num_layer):
            hidden_layers += [nn.Linear(last_size, hidden_size), nn.ReLU(hidden_size)]
            last_size = hidden_size
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(last_size, output_size)

    def forward(self, e_intent, e_price, e_pmask, e_extra=None):
        # (batch, emb_dim * state_num)
        old_et = e_intent
        e_intent = self.embeddings(e_intent).view(-1, self.embeddings.embedding_dim*self.state_length)
        if self.fix_emb:
            e_intent = e_intent.detach()
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


class PolicyDecoder(nn.Module):

    max_price, min_price = 2, -0.5

    def __init__(self, encoder_size, intent_size, num_layer=2, hidden_size=128):
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

    def forward(self, e_intent, e_price, e_pmask, e_extra=None):
        """
        """
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

    def forward(self, e_intent, e_price, e_pmask, e_extra=None):
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