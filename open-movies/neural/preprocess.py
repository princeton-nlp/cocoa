'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import time
import os
import numpy as np
from itertools import izip, izip_longest
from collections import namedtuple, defaultdict

from cocoa.core.util import read_pickle, write_pickle, read_json
from cocoa.core.entity import Entity, CanonicalEntity, is_entity
from cocoa.lib.bleu import compute_bleu
from cocoa.model.vocab import Vocabulary
from cocoa.model.trie import Trie

from core.tokenizer import tokenize
from batcher import DialogueBatcherFactory

def add_preprocess_arguments(parser):
    parser.add_argument('--entity-encoding-form', choices=['canonical', 'type'], default='canonical', help='Input entity form to the encoder')
    parser.add_argument('--entity-decoding-form', choices=['canonical', 'type'], default='canonical', help='Input entity form to the decoder')
    parser.add_argument('--entity-target-form', choices=['canonical', 'type'], default='canonical', help='Output entity form to the decoder')
    parser.add_argument('--candidates-path', nargs='*', default=[], help='Path to json file containing retrieved candidates for dialogues')
    parser.add_argument('--cache', default='.cache', help='Path to cache for preprocessed batches')
    parser.add_argument('--ignore-cache', action='store_true', help='Ignore existing cache')
    parser.add_argument('--mappings', help='Path to vocab mappings')

SpecialSymbols = namedtuple('SpecialSymbols', ['EOS', 'GO', 'DONE', 'PAD'])
markers = SpecialSymbols(EOS='</s>', GO='<go>', DONE='<done>', PAD='<pad>')

def build_vocab(dialogues, special_symbols=[], entity_forms=[]):
    vocab = Vocabulary(offset=0, unk=True)

    def _add_entity(entity):
        for entity_form in entity_forms:
            word = Preprocessor.get_entity_form(entity, entity_form)
            vocab.add_word(word)

    # Add words
    for dialogue in dialogues:
        assert dialogue.is_int is False
        for turn in dialogue.token_turns:
            for token in turn:
                if is_entity(token):
                    _add_entity(token)
                else:
                    vocab.add_word(token)

    # Add special symbols
    vocab.add_words(special_symbols, special=True)
    vocab.finish(size_threshold=10000)
    print 'Utterance vocabulary size:', vocab.size
    return vocab

def create_mappings(dialogues, schema, entity_forms):
    vocab = build_vocab(dialogues, markers, entity_forms)
    return {'vocab': vocab,
            }

class TextIntMap(object):
    '''
    Map between text and int for visualizing results.
    '''
    def __init__(self, vocab, preprocessor):
        self.vocab = vocab
        self.entity_forms = preprocessor.entity_forms
        self.preprocessor = preprocessor

    def pred_to_input(self, preds):
        '''
        Convert decoder outputs to decoder inputs.
        '''
        if self.entity_forms['target'] == self.entity_forms['decoding']:
            return preds
        preds_utterances = [self.int_to_text(pred) for pred in preds]
        input_utterances = [self.preprocessor.process_utterance(utterance, 'decoding') for utterance in preds_utterances]
        inputs = np.array([self.text_to_int(utterance, 'decoding') for utterance in input_utterances])
        return inputs

    def text_to_int(self, utterance, stage=None):
        '''
        Process entities in the utterance based on whether it is used for encoding, decoding
        or ground truth.
        '''
        tokens = self.preprocessor.process_utterance(utterance, stage)
        return [self.vocab.to_ind(token) for token in tokens]

    def int_to_text(self, inds, stage=None):
        '''
        Inverse of text_to_int.
        '''
        toks = [self.vocab.to_word(ind) for ind in inds]
        return toks

class Dialogue(object):
    textint_map = None
    ENC = 0
    DEC = 1
    TARGET = 2
    num_stages = 3  # encoding, decoding, target
    num_context = 1

    def __init__(self, agent, kb, uuid):
        '''
        Dialogue data that is needed by the model.
        '''
        self.uuid = uuid
        self.agent = agent
        self.kb = kb
        # token_turns: tokens and entitys (output of entitylink)
        self.token_turns = []
        # turns: input tokens of encoder, decoder input and target, later converted to integers
        self.turns = [[], [], []]
        self.agents = []
        self.is_int = False  # Whether we've converted it to integers

        self.token_candidates = None
        self.candidates = None
        self.true_candidate_inds = None

    @property
    def num_turns(self):
        return len(self.turns[0])

    #def join_turns(self):
    #    all_turns = []
    #    role_to_id = {'buyer': int_markers.GO_B, 'seller': int_markers.GO_S}
    #    for agent, turn in izip(self.agents, self.turns[0]):
    #        start_symbol = role_to_id[self.agent_to_role[agent]]
    #        all_turns.append([start_symbol] + turn)
    #    all_tokens = [x for turn in all_turns for x in turn]
    #    return all_tokens, all_turns

    def num_tokens(self):
        return sum([len(t) for t in self.token_turns])

    def add_ground_truth(self, candidates, token_turns):
        true_candidate_inds = []
        for cands, turn in izip(candidates, token_turns):
            if not cands:
                inds = []
            else:
                inds = []
                for i, cand in enumerate(cands):
                    if cand == turn:
                        inds.append(i)
                if len(inds) == 0:
                    cands.insert(0, turn)
                    del cands[-1]
                    inds.append(0)
            true_candidate_inds.append(inds)
        return candidates, true_candidate_inds

    def add_candidates(self, candidates, add_ground_truth=False):
        assert len(candidates) == len(self.token_turns)
        if add_ground_truth:
            candidates, self.true_candidate_inds = self.add_ground_truth(candidates, self.token_turns)
        self.token_candidates = candidates

    def add_utterance(self, agent, utterances):
        # Always start from the partner agent
        if len(self.agents) == 0 and agent == self.agent:
            self._add_utterance(1 - self.agent, [])
        self._add_utterance(agent, utterances)

    def _insert_markers(self, agent, utterance, new_turn):
        # Mark end of sentence
        utterance.append(markers.EOS)

        # Insert GO
        if new_turn:
            start_symbol = markers.GO
            utterance.insert(0, start_symbol)

        return utterance

    def _add_utterance(self, agent, utterance):
        # Same agent talking
        if len(self.agents) > 0 and agent == self.agents[-1]:
            new_turn = False
        else:
            new_turn = True

        utterance = self._insert_markers(agent, utterance, new_turn)
        entities = [x if is_entity(x) else None for x in utterance]

        if not new_turn:
            self.token_turns[-1].extend(utterance)
        else:
            self.agents.append(agent)
            self.token_turns.append(utterance)

    def candidates_to_int(self):
        self.candidates = []
        for turn_candidates in self.token_candidates:
            if turn_candidates is None:
                # Encoding turn
                self.candidates.append(None)
            else:
                c = [self.textint_map.text_to_int(c, 'decoding') for c in turn_candidates]
                self.candidates.append(c)

    def convert_to_int(self):
        if self.is_int:
            return

        for turn in self.token_turns:
            for turns, stage in izip(self.turns, ('encoding', 'decoding', 'target')):
                turns.append(self.textint_map.text_to_int(turn, stage))

        if self.token_candidates:
            self.candidates_to_int()

        self.is_int = True

    def _pad_list(self, l, size, pad):
        for i in xrange(len(l), size):
            l.append(pad)
        return l

    def pad_turns(self, num_turns):
        '''
        Pad turns to length num_turns.
        '''
        self.agents = self._pad_list(self.agents, num_turns, None)
        for turns in self.turns:
            self._pad_list(turns, num_turns, [])
        if self.candidates:
            self.candidates = self._pad_list(self.candidates, num_turns, [])

class Preprocessor(object):
    '''
    Preprocess raw utterances: tokenize, entity linking.
    Convert an Example into a Dialogue data structure used by DataGenerator.
    '''
    def __init__(self, schema, lexicon, entity_encoding_form, entity_decoding_form, entity_target_form):
        self.attributes = schema.attributes
        self.attribute_types = schema.get_attributes()
        self.lexicon = lexicon
        self.entity_forms = {'encoding': entity_encoding_form,
                'decoding': entity_decoding_form,
                'target': entity_target_form}

    @classmethod
    def get_entity_form(cls, entity, form):
        '''
        An entity is represented as (surface_form, (canonical_form, type)).
        '''
        assert len(entity) == 2
        if form == 'surface':
            return entity.surface
        elif form == 'type':
            return '<%s>' % entity.canonical.type
        elif form == 'canonical':
            return entity._replace(surface='')
        else:
            raise ValueError('Unknown entity form %s' % form)

    def process_utterance(self, utterance, stage=None):
        if stage is None:
            return [self.get_entity_form(x, 'canonical') if is_entity(x) else x for x in utterance]
        else:
            return [self.get_entity_form(x, self.entity_forms[stage]) if is_entity(x) else x for x in utterance]

    def _process_example(self, ex):
        '''
        Convert example to turn-based dialogue from each agent's perspective
        Create two Dialogue objects for each example
        '''
        kbs = ex.scenario.kbs
        for agent in (0, 1):
            dialogue = Dialogue(agent, kbs[agent], ex.ex_id)
            for e in ex.events:
                utterance = self.process_event(e, dialogue.kb)
                if utterance:
                    dialogue.add_utterance(e.agent, utterance)
            yield dialogue

    def process_event(self, e, kb):
        '''
        Tokenize, link entities
        '''
        if e.action == 'message':
            # Lower, tokenize, link entity
            #entity_tokens = self.lexicon.link_entity(tokenize(e.data))
            entity_tokens = tokenize(e.data)
            if entity_tokens:
                return entity_tokens
            else:
                return None
        elif e.action == 'done':
            entity_tokens = [markers.DONE]
            return entity_tokens
        else:
            raise ValueError('Unknown event action.')

    @classmethod
    def skip_example(cls, example):
        tokens = {0: 0, 1: 0}
        turns = {0: 0, 1: 0}
        for event in example.events:
            if event.action == "message":
                msg_tokens = tokenize(event.data)
                tokens[event.agent] += len(msg_tokens)
                turns[event.agent] += 1
        if tokens[0] < 40 and tokens[1] < 40:
            return True
        if turns[0] < 2 or turns[1] < 2:
            return True
        return False

    def preprocess(self, examples):
        dialogues = []
        for ex in examples:
            if self.skip_example(ex):
                continue
            for d in self._process_example(ex):
                dialogues.append(d)
        return dialogues

class DataGenerator(object):
    # TODO: hack
    trie = None

    def __init__(self, train_examples, dev_examples, test_examples, preprocessor,
            args, schema, mappings=None, retriever=None, cache='.cache',
            ignore_cache=False, candidates_path=[], num_context=1, batch_size=1,
            trie_path=None, model_config={}, add_ground_truth=True):
        examples = {'train': train_examples, 'dev': dev_examples, 'test': test_examples}
        self.num_examples = {k: len(v) if v else 0 for k, v in examples.iteritems()}

        # Build retriever given training dialogues
        self.retriever = retriever

        self.cache = cache
        self.ignore_cache = ignore_cache
        if (not os.path.exists(cache)) or ignore_cache:
            if retriever is not None:
                self.cached_candidates = self.retriever.load_candidates(candidates_path)
                print 'Cached candidates for %d dialogues' % len(self.cached_candidates)

            # NOTE: each dialogue is made into two examples from each agent's perspective
            self.dialogues = {k: preprocessor.preprocess(v)  for k, v in examples.iteritems() if v}

            for fold, dialogues in self.dialogues.iteritems():
                print '%s: %d dialogues out of %d examples' % (fold, len(dialogues), self.num_examples[fold])
        else:
            self.dialogues = {k: None  for k, v in examples.iteritems() if v}
            print 'Using cached data from', cache

        if not mappings:
            mappings = create_mappings(self.dialogues['train'], schema, preprocessor.entity_forms.values())
        self.mappings = mappings

        self.textint_map = TextIntMap(mappings['vocab'], preprocessor)
        Dialogue.mappings = mappings
        Dialogue.textint_map = self.textint_map
        Dialogue.preprocessor = preprocessor
        Dialogue.num_context = num_context

        global int_markers
        int_markers = SpecialSymbols(*[mappings['vocab'].to_ind(m) for m in markers])

        self.dialogue_batcher = DialogueBatcherFactory.get_dialogue_batcher(model_config, int_markers=int_markers)

        self.batches = {}
        for k, dialogues in self.dialogues.iteritems():
            _batches = self.create_batches(k, dialogues, batch_size, args.verbose, add_ground_truth=add_ground_truth)
            self.batches[k] = _batches

        self.trie = None
        # NOTE: Trie should be built after batches are created
        #self.trie = self.create_trie(self.batches.get('train', None), trie_path)
        #for name, batches in self.batches.iteritems():
        #    for batch in batches:
        #        for b in batch['batch_seq']:
        #            b['mask'] = self.get_mask(b['targets'], name)

    def get_mask(self, decoder_targets, split):
        batch_size, seq_len = decoder_targets.shape
        mask = np.zeros([batch_size, seq_len, self.mappings['vocab'].size], dtype=np.bool)
        for batch_id, targets in enumerate(decoder_targets):
            for time_step, t in enumerate(targets):
                prefix = tuple(targets[:time_step][-5:])
                try:
                    allowed = self.trie.get_children(prefix)
                    if split == 'train':
                        if t != int_markers.PAD:
                            assert t in allowed
                    mask[batch_id, time_step, allowed] = True
                except KeyError:
                    mask[batch_id, time_step, :] = True
        return mask

    def convert_to_int(self):
        '''
        Convert tokens to integers.
        '''
        for fold, dialogues in self.dialogues.iteritems():
            for dialogue in dialogues:
                dialogue.convert_to_int()

    def get_dialogue_batch(self, dialogues):
        return DialogueBatcher(dialogues).create_batch()

    def dialogue_sort_score(self, d):
        # Sort dialogues by number o turns
        return len(d.turns[0])

    def create_dialogue_batches(self, dialogues, batch_size):
        dialogue_batches = []
        dialogues.sort(key=lambda d: self.dialogue_sort_score(d))
        N = len(dialogues)
        start = 0
        while start < N:
            # NOTE: last batch may have a smaller size if we don't have enough examples
            end = min(start + batch_size, N)
            dialogue_batch = dialogues[start:end]
            dialogue_batches.append(self.dialogue_batcher.create_batch(dialogue_batch))
            start = end
        return dialogue_batches

    def get_all_responses(self, name):
        dialogues = self.dialogues[name]
        responses = {'seller': [], 'buyer': []}
        for dialogue in dialogues:
            for turn, role in izip(dialogue.token_turns, dialogue.roles):
                responses[role].extend(turn)
        return responses

    def rewrite_candidate(self, fillers, candidate):
        rewritten = []
        tokens = candidate
        if not tokens:
            return rewritten
        for i, tok in enumerate(tokens):
            if is_entity(tok) and tok.canonical.type == 'slot':
                for filler in fillers:
                    ss = filler.split()
                    new_tokens = list(tokens)
                    del new_tokens[i]
                    for j, s in enumerate(ss):
                        new_tokens.insert(i+j, Entity(s, CanonicalEntity('', 'slot')))
                    new_cand = new_tokens
                    rewritten.append(new_cand)
        return rewritten

    def create_batches(self, name, dialogues, batch_size, verbose, add_ground_truth=True):
        if not os.path.isdir(self.cache):
            os.makedirs(self.cache)
        cache_file = os.path.join(self.cache, '%s_batches.pkl' % name)
        if (not os.path.exists(cache_file)) or self.ignore_cache:
            if self.retriever is not None:
                for dialogue in dialogues:
                    k = (dialogue.uuid, dialogue.role)
                    # candidates: list of list containing num_candidates responses for each turn of the dialogue
                    # None candidates means that it is not the agent's speaking turn
                    if k in self.cached_candidates:
                        candidates = self.cached_candidates[k]
                    else:
                        candidates = self.retriever.retrieve_candidates(dialogue, json_dict=False)


                    dialogue.add_candidates(candidates, add_ground_truth=add_ground_truth)
                self.retriever.report_search_time()

            for dialogue in dialogues:
                dialogue.convert_to_int()

            dialogue_batches = self.create_dialogue_batches(dialogues, batch_size)
            print 'Write %d batches to cache %s' % (len(dialogue_batches), cache_file)
            start_time = time.time()
            write_pickle(dialogue_batches, cache_file)
            print '[%d s]' % (time.time() - start_time)
        else:
            start_time = time.time()
            dialogue_batches = read_pickle(cache_file)
            if verbose:
                print 'Read %d batches from cache %s' % (len(dialogue_batches), cache_file)
                print '[%d s]' % (time.time() - start_time)
        return dialogue_batches

    def generator(self, name, shuffle=True):
        dialogue_batches = self.batches[name]
        #yield len(dialogue_batches)
        inds = range(len(dialogue_batches))
        #while True:
        if shuffle:
            random.shuffle(inds)
        for ind in inds:
            yield dialogue_batches[ind]

    def create_trie(self, batches, path):
        if path is None:
            return None
        def seq_iter(batches):
            for batch in batches:
                for b in batch['batch_seq']:
                    targets = b['targets']
                    for target in targets:
                        yield target
        if not os.path.exists(path):
            trie = Trie()
            print 'Build prefix trie of length', trie.max_prefix_len
            start_time = time.time()
            trie.build_trie(seq_iter(batches))
            print '[%d s]' % (time.time() - start_time)
            print 'Write trie to', path
            write_pickle(trie, path)
        else:
            print 'Read trie from', path
            trie = read_pickle(path)
        return trie

class EvalDialogue(Dialogue):
    def __init__(self, agent, kb, uuid):
        super(EvalDialogue, self).__init__(agent, kb, uuid)
        self.candidate_scores = None

    def context_len(self):
        return sum([len(t) for t in self.token_turns[:-1]])

    def pad_turns(self, num_turns):
        '''
        Pad turns to length num_turns.
        '''
        self.agents = self._pad_list(self.agents, num_turns, None)
        for turns in self.turns:
            self._pad_list(turns, num_turns, [])

    def _pad_list(self, l, size, pad):
        # NOTE: for dialogues without enough turns/context, we need to pad at the beginning, the last utterance is the target
        for i in xrange(len(l), size):
            l.insert(0, pad)
        return l

class EvalDataGenerator(DataGenerator):
    def __init__(self, examples, preprocessor, mappings, num_context=1):
        self.dialogues = {'eval': [self.process_example(ex) for ex in examples]}
        self.num_examples = {k: len(v) if v else 0 for k, v in self.dialogues.iteritems()}
        print '%d eval examples' % self.num_examples['eval']

        self.mappings = mappings
        self.textint_map = TextIntMap(mappings['vocab'], preprocessor)

        EvalDialogue.mappings = mappings
        EvalDialogue.textint_map = self.textint_map
        EvalDialogue.preprocessor = preprocessor
        EvalDialogue.num_context = num_context

        global int_markers
        int_markers = SpecialSymbols(*[mappings['vocab'].to_ind(m) for m in markers])

    @classmethod
    def tuple_to_entity(cls, token):
        if isinstance(token, list):
            return Entity(token[0], CanonicalEntity(*token[1]))
        else:
            return token

    # TODO: move this to EvalExample, do it during reading
    def _process_utterance(self, utterance):
        '''
        Make sure the utterance matches preprocessor processed utterance.
        '''
        # Convert list in json file to Entity
        # Remove <go> at the beginning and </s> at the end
        # because they will be added in _add_utterance
        utterance = [self.tuple_to_entity(x) for x in utterance[1:-1]]
        return utterance

    def process_example(self, ex):
        d = EvalDialogue(ex.agent, ex.kb, ex.ex_id)
        for role, utterance in izip(ex.prev_roles, ex.prev_turns):
            agent = ex.agent if role == ex.role else (1 - ex.agent)
            d.add_utterance(agent, self._process_utterance(utterance))
        d.add_utterance(ex.agent, self._process_utterance(ex.target))

        for c in ex.candidates:
            if 'response' in c:
                c['response'] = self._process_utterance(c['response'])
        d.token_candidates = [ex.candidates]

        d.candidate_scores = ex.scores
        return d

    def dialogue_sort_score(self, d):
        # Sort dialogues by number o turns
        return d.context_len()

    def get_dialogue_batch(self, dialogues):
        return EvalDialogueBatcher(dialogues).create_batch()

    # TODO: new interface: create_batches
    def generator(self, split, batch_size, shuffle=True):
        dialogues = self.dialogues['eval']

        for dialogue in dialogues:
            dialogue.convert_to_int()
        dialogue_batches = self.create_dialogue_batches(dialogues, batch_size)

        yield len(dialogue_batches)
        inds = range(len(dialogue_batches))
        while True:
            if shuffle:
                random.shuffle(inds)
            for ind in inds:
                yield dialogue_batches[ind]

class LMDataGenerator(DataGenerator):
    def create_dialogue_batches(self, dialogues, batch_size, bptt_steps=35):
        dialogue_batches = []
        # TODO: only need diaogue from one direction
        dialogues.sort(key=lambda d: d.num_tokens())
        N = len(dialogues)
        start = 0
        while start < N:
            # NOTE: last batch may have a smaller size if we don't have enough examples
            end = min(start + batch_size, N)
            dialogue_batch = dialogues[start:end]
            dialogue_batches.append(LMDialogueBatcher(dialogue_batch).create_batch(bptt_steps))
            start = end
        return dialogue_batches

