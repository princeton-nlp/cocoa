import numpy as np
from itertools import zip_longest

import torch
from torch.autograd import Variable

from .symbols import markers

def pad_list_to_array(l, fillvalue, dtype):
    '''
    l: list of lists with unequal length
    return: np array with minimal padding
    '''
    return np.array(list(zip_longest(*l, fillvalue=fillvalue)), dtype=dtype).T

class Batch(object):
    def __init__(self, encoder_args, decoder_args, context_data, vocab,
                time_major=True, num_context=None, cuda=False, for_value=False):
        self.vocab = vocab
        self.num_context = num_context
        self.encoder_intent = encoder_args['intent']
        self.encoder_price = encoder_args['price']
        self.encoder_pmask = encoder_args['price_mask']

        self.for_value = for_value

        if not for_value:
            self.target_intent = decoder_args['intent']
            self.target_price = decoder_args['price']
            self.target_pmask = decoder_args['price_mask']
            # TODO: Get policy mask from the intent
            self.policy_mask = np.ones((len(self.encoder_intent), len(self.vocab)))
            offer_idx = self.vocab.to_ind(markers.OFFER)
            acc_idx = self.vocab.to_ind(markers.ACCEPT)
            rej_idx = self.vocab.to_ind(markers.REJECT)
            for i in range(len(self.encoder_intent)):
                if self.encoder_intent[i] == offer_idx:
                    self.policy_mask[i, :] = 0
                    self.policy_mask[i,[acc_idx, rej_idx]] = 1
        else:
            self.target_value = decoder_args['value']

        self.title_inputs = decoder_args['context']['title']
        self.desc_inputs = decoder_args['context']['description']

        self.size = len(self.encoder_intent)
        self.context_data = context_data

        unsorted_attributes = ['encoder_intent', 'encoder_price',
                               'title_inputs', 'desc_inputs']
        batch_major_attributes = ['encoder_intent', 'decoder_price', 'title_inputs', 'desc_inputs']

        if not for_value:
            unsorted_attributes += ['target_intent', 'target_price']
            batch_major_attributes += ['target_intent', 'target_price']
        else:
            unsorted_attributes += ['target_value']
            batch_major_attributes += ['target_value']

        # if num_context > 0:
        #     self.context_inputs = encoder_args['context'][0]
        #     unsorted_attributes.append('context_inputs')
        #     batch_major_attributes.append('context_inputs')

        # self.lengths, sorted_ids = self.sort_by_length(self.encoder_inputs)
        # self.tgt_lengths, _ = self.sort_by_length(self.decoder_inputs)

        # if time_major:
        #     for attr in batch_major_attributes:
        #         print(attr)
        #         if len(getattr(self, attr)) > 0:
        #             setattr(self, attr, np.swapaxes(getattr(self, attr), 0, 1))

        # To tensor/variable
        self.encoder_intent = self.to_variable(self.encoder_intent, 'long', cuda).unsqueeze(1)
        self.encoder_price = self.to_variable(self.encoder_price, 'float', cuda).unsqueeze(1)
        self.encoder_pmask = self.to_variable(self.encoder_pmask, 'float', cuda).unsqueeze(1)

        if not for_value:
            # print('ti0, ', self.target_intent)
            self.target_intent = self.to_variable(self.target_intent, 'long', cuda)
            self.target_price = self.to_variable(self.target_price, 'float', cuda)
            self.target_pmask = self.to_variable(self.target_pmask, 'float', cuda)
            # print('ti1, ', self.target_intent)
            for v in ['target_intent', 'target_price', 'target_pmask']:
                while True:
                    d = len(getattr(self, v).shape)
                    if d >= 2:
                        break
                    setattr(self, v, getattr(self, v).unsqueeze(d))
            # print('ti2, ', self.target_intent)

        else:
            self.target_value = self.to_variable(self.target_value, 'float', cuda).unsqueeze(1)

        self.title_inputs = self.to_variable(self.title_inputs, 'long', cuda)
        self.desc_inputs = self.to_variable(self.desc_inputs, 'long', cuda)
        # self.targets = self.to_variable(self.targets, 'long', cuda)
        # self.lengths = self.to_tensor(self.lengths, 'long', cuda)
        # self.tgt_lengths = self.to_tensor(self.tgt_lengths, 'long', cuda)
        # if num_context > 0:
        #     self.context_inputs = self.to_variable(self.context_inputs, 'long', cuda)

    @classmethod
    def to_tensor(cls, data, dtype, cuda=False):
        if type(data) == np.ndarray:
            data = data.tolist()
        if dtype == "long":
            tensor = torch.LongTensor(data)
        elif dtype == "float":
            tensor = torch.FloatTensor(data)
        else:
            raise ValueError
        return tensor.cuda() if cuda else tensor

    @classmethod
    def to_variable(cls, data, dtype, cuda=False):
        tensor = cls.to_tensor(data, dtype)
        var = Variable(tensor)
        return var.cuda() if cuda else var

    def sort_by_length(self, inputs):
        """
        Args:
            inputs (numpy.ndarray): (batch_size, seq_length)
        """
        pad = self.vocab.word_to_ind[markers.PAD]
        def get_length(seq):
            for i, x in enumerate(seq):
                if x == pad:
                    return i
            return len(seq)
        lengths = [get_length(s) for s in inputs]
        # TODO: look into how it works for all-PAD seqs
        lengths = [l if l > 0 else 1 for l in lengths]
        sorted_id = np.argsort(lengths)[::-1]
        return lengths, sorted_id

    def order_by_id(self, inputs, ids):
        if ids is None:
            return inputs
        else:
            if type(inputs) is np.ndarray:
                if len(inputs) == 0: return inputs
                return inputs[ids, :]
            elif type(inputs) is list:
                return [inputs[i] for i in ids]
            else:
                raise ValueError('Unknown input type {}'.format(type(inputs)))


class DialogueBatcher(object):
    def __init__(self, kb_pad=None, mappings=None, model='seq2seq', num_context=2):
        self.pad = mappings['utterance_vocab'].to_ind(markers.PAD)
        self.kb_pad = kb_pad
        self.mappings = mappings
        self.model = model
        self.num_context = num_context

    def _normalize_dialogue(self, dialogues):
        '''
        All dialogues in a batch should have the same number of turns.
        '''
        max_num_turns = max([d.num_lfs for d in dialogues])
        for dialogue in dialogues:
            dialogue.pad_turns(max_num_turns)
        num_turns = dialogues[0].num_turns
        return num_turns

    def _get_turn_batch_at(self, dialogues, STAGE, i):
        # For RL setting, we use lf as turns.
        if STAGE == 0:
            # encoder pad
            pad = self.mappings['src_vocab'].to_ind(markers.PAD)
        else:
            # decoder & target pad
            pad = self.mappings['tgt_vocab'].to_ind(markers.PAD)
        if i is None:
            # Return all turns
            tmp = [self._get_turn_batch_at(dialogues, STAGE, i) for i in range(dialogues[0].num_turns)]
            turns = {'intent': [], 'price': []}
            for k in turns:
                for j in tmp:
                    turns[k].append(j[k])
            return turns
        else:
            intent = []
            price = []
            for d in dialogues:
                tmp = d.lfs[i]
                intent.append(tmp['intent'])
                price.append(tmp.get('price'))
            turns = {'intent': intent, 'price': price}
            return turns

    def create_context_batch(self, dialogues, pad):
        category_batch = np.array([d.category for d in dialogues], dtype=np.int32)
        # TODO: make sure pad is consistent
        #pad = Dialogue.mappings['kb_vocab'].to_ind(markers.PAD)
        title_batch = pad_list_to_array([d.title for d in dialogues], pad, np.int32)
        # TODO: hacky: handle empty description
        description_batch = pad_list_to_array([[pad] if not d.description else d.description for d in dialogues], pad, np.int32)
        return {
                'category': category_batch,
                'title': title_batch,
                'description': description_batch,
                }

    def _get_agent_batch_at(self, dialogues, i):
        return [dialogue.agents[i] for dialogue in dialogues]

    def _get_kb_batch(self, dialogues):
        return [dialogue.kb for dialogue in dialogues]

    def _remove_last(self, array, value, pad):
        array = np.copy(array)
        nrows, ncols = array.shape
        for i in range(nrows):
            for j in range(ncols-1, -1, -1):
                if array[i][j] == value:
                    array[i][j] = pad
                    break
        return array

    def _remove_prompt(self, input_arr):
        '''
        Remove starter symbols (e.g. <go>) used for the decoder.
        input_arr: (batch_size, seq_len)
        '''
        # TODO: depending on prompt length
        return input_arr[:, 1:]

    def get_encoder_inputs(self, encoder_turns):
        intent = encoder_turns['intent']
        price = [p if p is not None else 0 for p in encoder_turns['price']]
        price_mask = [1 if p is not None else 0 for p in encoder_turns['price']]
        intent = np.array(intent, dtype=np.int32)
        price = np.array(price, dtype=np.float)
        price_mask = np.array(price_mask, dtype=np.int32)
        return intent, price, price_mask

    def get_encoder_context(self, encoder_turns, num_context):
        # |num_context| utterances before the last partner utterance
        encoder_context = [self._remove_prompt(turn) for turn in encoder_turns[-1*(num_context+1):-1]]
        if len(encoder_context) < num_context:
            batch_size = encoder_turns[0].shape[0]
            empty_context = np.full([batch_size, 1], self.pad, np.int32)
            for i in range(num_context - len(encoder_context)):
                encoder_context.insert(0, empty_context)
        return encoder_context

    def make_decoder_inputs_and_targets(self, decoder_turns, target_turns=None):
        intent = decoder_turns['intent']
        price = [p if p is not None else 0 for p in decoder_turns['price']]
        price_mask = [1 if p is not None else 0 for p in decoder_turns['price']]
        intent = np.array(intent, dtype=np.int32)
        price = np.array(price, dtype=np.float)
        price_mask = np.array(price_mask, dtype=np.int32)
        return intent, price, price_mask

    def _create_one_batch(self, encoder_turns=None, decoder_turns=None,
            target_turns=None, agents=None, uuids=None, kbs=None, kb_context=None,
            num_context=None, encoder_tokens=None, decoder_tokens=None):
        # encoder_context = self.get_encoder_context(encoder_turns, num_context)

        # print('encoder_turns: ', encoder_turns)
        encoder_intent, encoder_price, encoder_price_mask = self.get_encoder_inputs(encoder_turns)
        target_intent, target_price, target_price_mask = self.make_decoder_inputs_and_targets(decoder_turns, target_turns)

        encoder_args = {
                'intent': encoder_intent,
                'price': encoder_price,
                'price_mask': encoder_price_mask,
                # 'context': encoder_context,
                }
        decoder_args = {
                'intent': target_intent,
                'price': target_price,
                'price_mask': target_price_mask,
                'context': kb_context,
                }
        context_data = {
                'encoder_tokens': encoder_tokens,
                'decoder_tokens': decoder_tokens,
                'agents': agents,
                'kbs': kbs,
                'uuids': uuids,
                }
        batch = {
                'encoder_args': encoder_args,
                'decoder_args': decoder_args,
                'context_data': context_data,
                }
        return batch

    def int_to_text(self, array, textint_map, stage):
        tokens = [str(x) for x in textint_map.int_to_text((x for x in array if x != self.pad), stage)]
        return ' '.join(tokens)

    def list_to_text(self, tokens):
        return ' '.join(str(x) for x in tokens)

    def print_batch(self, batch, example_id, textint_map, preds=None):
        i = example_id
        print('-------------- Example {} ----------------'.format(example_id))
        if len(batch['decoder_tokens'][i]) == 0:
            print('PADDING')
            return False
        print('RAW INPUT:\n {}'.format(self.list_to_text(batch['encoder_tokens'][i])))
        print('RAW TARGET:\n {}'.format(self.list_to_text(batch['decoder_tokens'][i])))
        print('ENC INPUT:\n {}'.format(self.int_to_text(batch['encoder_args']['intent'][i], textint_map, 'encoding')))
        print('DEC INPUT:\n {}'.format(self.int_to_text(batch['decoder_args']['intent'][i], textint_map, 'decoding')))
        #print('TARGET:\n {}'.format(self.int_to_text(batch['decoder_args']['targets'][i], textint_map, 'target')))
        if preds is not None:
            print('PRED:\n {}'.format(self.int_to_text(preds[i], textint_map, 'target')))
        return True

    def _get_token_turns_at(self, dialogues, i):
        stage = 0
        if not hasattr(dialogues[0], 'token_turns'):
            return None
        # Return None for padded turns
        return [dialogue.token_turns[i] if i < len(dialogue.token_turns) else ''
                for dialogue in dialogues]

    def _get_dialogue_data(self, dialogues):
        '''
        Data at the dialogue level, i.e. same for all turns.
        '''
        agents = self._get_agent_batch_at(dialogues, 1)  # Decoding agent
        kbs = self._get_kb_batch(dialogues)
        uuids = [d.uuid for d in dialogues]
        kb_context_batch = self.create_context_batch(dialogues, self.kb_pad)
        return {
                'agents': agents,
                'kbs': kbs,
                'uuids': uuids,
                'kb_context': kb_context_batch,
                }

    def get_encoding_turn_ids(self, num_turns):
        # NOTE: when creating dialogue turns (see add_utterance), we have set the first utterance to be from the encoding agent
        encode_turn_ids = range(0, num_turns-1, 2)
        return encode_turn_ids

    def _get_lf_batch_at(self, dialogues, i):
        pad = self.mappings['lf_vocab'].to_ind(markers.PAD)
        return pad_list_to_array([d.lfs[i] for d in dialogues], pad, np.int32)

    def create_batch(self, dialogues):
        num_turns = self._normalize_dialogue(dialogues)
        # print('num turns: ', num_turns)
        dialogue_data = self._get_dialogue_data(dialogues)

        dialogue_class = type(dialogues[0])
        ENC, DEC, TARGET = dialogue_class.ENC, dialogue_class.DEC, dialogue_class.TARGET

        encode_turn_ids = self.get_encoding_turn_ids(num_turns)
        encoder_turns_all = self._get_turn_batch_at(dialogues, ENC, None)

        # NOTE: encoder_turns contains all previous dialogue context, |num_context|
        # decides how many turns to use
        batch_seq = [
            self._create_one_batch(
                # encoder_turns=encoder_turns_all[:i+1],
                encoder_turns=self._get_turn_batch_at(dialogues, ENC, i),
                decoder_turns=self._get_turn_batch_at(dialogues, DEC, i+1),
                target_turns=self._get_turn_batch_at(dialogues, TARGET, i+1),
                encoder_tokens=self._get_token_turns_at(dialogues, i),
                decoder_tokens=self._get_token_turns_at(dialogues, i+1),
                agents=dialogue_data['agents'],
                uuids=dialogue_data['uuids'],
                kbs=dialogue_data['kbs'],
                kb_context=dialogue_data['kb_context'],
                num_context=self.num_context,
                )
                for i in encode_turn_ids
            ]

        # bath_seq: A sequence of batches that can be processed in turn where
        # the state of each batch is passed on to the next batch
        return batch_seq

    def create_batch_critic(self, dialogues):
        num_turns = self._normalize_dialogue(dialogues)
        dialogue_data = self._get_dialogue_data(dialogues)

        dialogue_class = type(dialogues[0])
        ENC, DEC, TARGET = dialogue_class.ENC, dialogue_class.DEC, dialogue_class.TARGET

        encode_turn_ids = self.get_encoding_turn_ids(num_turns)
        encoder_turns_all = self._get_turn_batch_at(dialogues, ENC, None)

        # NOTE: encoder_turns contains all previous dialogue context, |num_context|
        # decides how many turns to use
        batch_seq = [
            dialogues[0].
            self._create_one_batch(
                encoder_turns=encoder_turns_all[:i + 1],
                decoder_turns=self._get_turn_batch_at(dialogues, DEC, i + 1),
                target_turns=self._get_turn_batch_at(dialogues, TARGET, i + 1),
                encoder_tokens=self._get_token_turns_at(dialogues, i),
                decoder_tokens=self._get_token_turns_at(dialogues, i + 1),
                agents=dialogue_data['agents'],
                uuids=dialogue_data['uuids'],
                kbs=dialogue_data['kbs'],
                kb_context=dialogue_data['kb_context'],
                num_context=self.num_context,
            )
            for i in encode_turn_ids
        ]

        # bath_seq: A sequence of batches that can be processed in turn where
        # the state of each batch is passed on to the next batch
        return batch_seq

class DialogueBatcherWrapper(object):
    def __init__(self, batcher):
        self.batcher = batcher
        # TODO: fix kb_pad, hacky
        self.kb_pad = batcher.kb_pad

    def create_batch(self, dialogues):
        raise NotImplementedError

    def create_context_batch(self, dialogues, pad):
        return self.batcher.create_context_batch(dialogues, pad)

    def get_encoder_inputs(self, encoder_turns):
        return self.batcher.get_encoder_inputs(encoder_turns)

    def get_encoder_context(self, encoder_turns, num_context):
        return self.batcher.get_encoder_context(encoder_turns, num_context)

    def list_to_text(self, tokens):
        return self.batcher.list_to_text(tokens)

    def _get_turn_batch_at(self, dialogues, STAGE, i):
        return self.batcher._get_turn_batch_at(dialogues, STAGE, i)


class DialogueBatcherFactory(object):
    @classmethod
    def get_dialogue_batcher(cls, model, **kwargs):
        if model in ('seq2seq', 'lf2lf', 'tom'):
            batcher = DialogueBatcher(**kwargs)
        else:
            raise ValueError
        return batcher
