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
from onmt.RLModels import StateEncoder, PolicyDecoder, PolicyModel
from onmt.Utils import use_gpu

from cocoa.io.utils import read_pickle
from neural import make_model_mappings


def make_embeddings(opt, word_dict, emb_length, for_encoder=True):
    return nn.Embedding(len(word_dict), emb_length)


def make_encoder(opt, embeddings, output_size):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    return StateEncoder(embeddings, output_size=output_size)


def make_decoder(opt, encoder_size, intent_size):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    return PolicyDecoder(encoder_size=encoder_size, intent_size=intent_size)


def load_test_model(model_path, opt, dummy_opt):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    mappings = read_pickle('{}/vocab.pkl'.format(model_opt.mappings))

    # mappings = read_pickle('{0}/{1}/vocab.pkl'.format(model_opt.mappings, model_opt.model))
    mappings = make_model_mappings(model_opt.model, mappings)

    model = make_base_model(model_opt, mappings, use_gpu(opt), checkpoint)
    model.eval()
    return mappings, model, model_opt


def make_base_model(model_opt, mappings, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    # Make encoder.
    src_dict = mappings['src_vocab']
    src_embeddings = make_embeddings(model_opt, src_dict, model_opt.word_vec_size)
    encoder = make_encoder(model_opt, src_embeddings, model_opt.hidden_size)
    print('encoder', encoder)

    # Make decoder.
    tgt_dict = mappings['tgt_vocab']
    tgt_embeddings = src_embeddings

    decoder = make_decoder(model_opt, model_opt.hidden_size, len(tgt_dict))
    print('decoder', decoder)


    model = PolicyModel(encoder, decoder)
    model.model_type = 'text'

    # Make Generator.

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
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
    else:
        model.cpu()

    return model