"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.RLModels import StateEncoder, UtteranceEncoder, StateUtteranceEncoder, \
    MeanEncoder, RNNEncoder, \
    PolicyDecoder, PolicyModel, ValueModel, ValueDecoder, \
    HistoryEncoder, CurrentEncoder, \
    HistoryModel, CurrentModel, \
    MixedPolicy, SinglePolicy
from onmt.Utils import use_gpu

from cocoa.io.utils import read_pickle
from neural import make_model_mappings


def make_embeddings(opt, word_dict, emb_length, for_encoder=True):
    return nn.Embedding(len(word_dict), emb_length)


def make_encoder(opt, embeddings, intent_size, output_size, use_history=False, fix_emb=False, ):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    # encoder = StateEncoder(intent_size=intent_size, output_size=output_size,
    #                     state_length=opt.state_length, extra_size=3 if opt.dia_num>0 else 0 )

    diaact_size = (intent_size+1)
    extra_size = 3 + 2
    hidden_size = opt.hidden_size
    if not opt.use_utterance:
        embeddings = None
    if use_history:
        encoder = HistoryEncoder(diaact_size*2, hidden_size, extra_size, embeddings, output_size)
    else:
        encoder = CurrentEncoder(diaact_size*opt.state_length+extra_size, embeddings, output_size)

    return encoder


def make_decoder(opt, encoder_size, intent_size, price_action=False, output_value=False):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if output_value:
        return SinglePolicy(encoder_size, 1, num_layer=2)
    if price_action:
        return MixedPolicy(encoder_size, intent_size, 4)
    return MixedPolicy(encoder_size, intent_size, 1)
    # return PolicyDecoder(encoder_size=encoder_size, intent_size=intent_size)


def load_test_model(model_path, opt, dummy_opt):
    if model_path is not None:
        print('Load model from {}.'.format(model_path))
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

        model_opt = checkpoint['opt']
        for arg in dummy_opt:
            if arg not in model_opt:
                model_opt.__dict__[arg] = dummy_opt[arg]
    else:
        print('Build model from scratch.')
        checkpoint = None
        model_opt = opt

    mappings = read_pickle('{}/vocab.pkl'.format(model_opt.mappings))

    # mappings = read_pickle('{0}/{1}/vocab.pkl'.format(model_opt.mappings, model_opt.model))
    mappings = make_model_mappings(model_opt.model, mappings)

    model, critic = make_base_model(model_opt, mappings, use_gpu(opt), checkpoint)
    model.eval()
    critic.eval()
    return mappings, model, model_opt, critic


def init_model(model, checkpoint, model_opt):
    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)


def make_sl_model(model_opt, mappings, gpu, checkpoint=None):
    intent_size = mappings['lf_vocab'].size

    # Make encoder.
    src_dict = mappings['utterance_vocab']
    src_embeddings = make_embeddings(model_opt, src_dict, model_opt.word_vec_size)
    encoder = make_encoder(model_opt, src_embeddings, intent_size, model_opt.hidden_size)
    # print('encoder', encoder)

    # Make decoder.
    tgt_dict = mappings['tgt_vocab']

    decoder = make_decoder(model_opt, model_opt.hidden_size, intent_size)
    # print('decoder', decoder)

    model = CurrentModel(encoder, decoder)

    init_model(model, checkpoint, model_opt)

    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model


def make_base_model(model_opt, mappings, gpu, checkpoint=None, type='sl'):
    """s
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    intent_size = mappings['lf_vocab'].size

    # Make encoder.
    src_dict = mappings['utterance_vocab']
    src_embeddings = make_embeddings(model_opt, src_dict, model_opt.word_vec_size)
    encoder = make_encoder(model_opt, src_embeddings, intent_size, model_opt.hidden_size)
    # print('encoder', encoder)

    # Make decoder.
    tgt_dict = mappings['tgt_vocab']


    decoder = make_decoder(model_opt, model_opt.hidden_size, intent_size)
    # print('decoder', decoder)

    model = CurrentModel(encoder, decoder)


    # Policy Model : Current Encoder + Intent | Price(action)
    model = PolicyModel(encoder, decoder)

    model.model_type = 'text'

    # Make Critic.
    # critic_embeddings = src_embeddings
    # critic_embeddings = make_embeddings(model_opt, src_dict, model_opt.word_vec_size)
    value_encoder = make_encoder(model_opt, src_embeddings, intent_size, model_opt.hidden_size, fix_emb=True)
    value_decoder = make_decoder(model_opt, model_opt.hidden_size, intent_size, output_value=True)
    critic = ValueModel(value_encoder, value_decoder)
    # model.critic = critic

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])

        # Get parameters of critic model
        # if hasattr(model, 'critic'):
        if critic is not None:
            if checkpoint.get('critic') is not None:
                print('Loading critic model parameters.')
                critic.load_state_dict(checkpoint['critic'])
            else:
                print('Intializing critic parameters.')
                for p in critic.parameters():
                    p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)

            # Get parameters of critic model
            if critic is not None:
                print('Intializing critic parameters.')
                for p in critic.parameters():
                    p.data.uniform_(-model_opt.param_init, model_opt.param_init)

        wordvec = {'utterance': model_opt.pretrained_wordvec[0]}
        if len(model_opt.pretrained_wordvec) > 1:
            wordvec['kb'] = model_opt.pretrained_wordvec[1]

        def load_wordvec(embeddings, name):
            embeddings.load_pretrained_vectors(
                    wordvec[name], model_opt.fix_pretrained_wordvec)

        # Don't need pretrained word vec for LFs
        if not model_opt.model in ('lf2lf',):
            load_wordvec(model.encoder.embeddings, 'utterance')
            if hasattr(model, 'context_embedder'):
                load_wordvec(model.context_embedder.embeddings, 'utterance')
        if hasattr(model, 'kb_embedder') and model.kb_embedder is not None:
            load_wordvec(model.kb_embedder.embeddings, 'kb')

        if model_opt.model == 'seq2seq':
            load_wordvec(model.decoder.embeddings, 'utterance')

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
        critic.cuda()
    else:
        model.cpu()
        critic.cpu()

    return model, critic


def make_critic_model(model_opt, mappings, gpu, encoder=None):
    # Make encoder.
    if encoder is None:
        src_dict = mappings['src_vocab']
        src_embeddings = make_embeddings(model_opt, src_dict, model_opt.word_vec_size)
        encoder = make_encoder(model_opt, src_embeddings, model_opt.hidden_size)
    # print('encoder', encoder)

    # Make decoder.
    tgt_dict = mappings['tgt_vocab']

    decoder = make_decoder(model_opt, model_opt.hidden_size, len(tgt_dict), output_value=True)
    # print('decoder', decoder)

    model = PolicyModel(encoder, decoder)

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model
