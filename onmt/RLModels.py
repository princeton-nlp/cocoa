import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq

from .Models import RNNEncoder


class StateEncoder(nn.Module):

    def __init__(self, embedding, output_size=64, num_layer=1, hidden_size=64):
        super(StateEncoder, self).__init__()
        self.embeddings = embedding

        last_size = embedding.embedding_dim + 1
        hidden_layers = []
        for i in range(num_layer):
            hidden_layers += [nn.Linear(last_size, hidden_size), nn.ReLU(hidden_size)]
            last_size = hidden_size
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(last_size, output_size)

    def forward(self, e_intent, e_price, e_pmask):
        e_intent = self.embeddings(e_intent).squeeze(1)
        # print(e_price.shape, e_pmask.shape)
        e_price = e_price.mul(e_pmask)
        # print(e_price.shape)

        e_output = torch.cat([e_intent, e_price], dim=1)
        # print(e_output.shape)
        e_output = self.hidden_layers(e_output)
        e_output = self.output_layer(e_output)

        return e_output

class PolicyDecoder(nn.Module):
    
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
        if self.training:
            p = self.reparameterize(p_mean, p_var)
        else:
            p = p_mean
        #
        # # TODO: test p_mean
        # p = p_mean

        return policy, p


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

    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(PolicyModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, e_intent, e_price, e_pmask):
        """
        """

        enc_final = self.encoder(e_intent, e_price, e_pmask)
        policy, price = self.decoder(enc_final)
        return policy, price

class ValueModel(nn.Module):
    """
        encoder + value
    """
    def __init__(self):
        pass
