import random
import re
import numpy as np
import torch
from onmt.Utils import use_gpu

from cocoa.model.vocab import Vocabulary
from cocoa.core.entity import is_entity, Entity

from core.event import Event
from .session import Session
from neural.preprocess import markers, Dialogue
from neural.batcher_rl import Batch

class NeuralSession(Session):
    def __init__(self, agent, kb, env):
        super(NeuralSession, self).__init__(agent)
        self.env = env
        self.kb = kb
        self.builder = env.utterance_builder
        self.generator = env.dialogue_generator
        self.cuda = env.cuda

        self.batcher = self.env.dialogue_batcher
        self.dialogue = Dialogue(agent, kb, None)
        self.dialogue.kb_context_to_int()
        self.kb_context_batch = self.batcher.create_context_batch([self.dialogue], self.batcher.kb_pad)
        self.max_len = 100

    # TODO: move this to preprocess?
    def convert_to_int(self):
        for i, turn in enumerate(self.dialogue.token_turns):
            for curr_turns, stage in zip(self.dialogue.turns, ('encoding', 'decoding', 'target')):
                if i >= len(curr_turns):
                    curr_turns.append(self.env.textint_map.text_to_int(turn, stage))

    def receive(self, event):
        if event.action in Event.decorative_events:
            return
        # print(event.data)
        # Parse utterance
        utterance = self.env.preprocessor.process_event(event, self.kb)
        # print('utterance is:', utterance)

        # Empty message
        if utterance is None:
            return

        #print 'receive:', utterance
        # self.dialogue.add_utterance(event.agent, utterance)
        # state = event.metadata.copy()
        state = {'enc_output': event.metadata['enc_output']}

        utterance_int = self.env.textint_map.text_to_int(utterance)
        state['action'] = utterance_int[0]
        self.dialogue.add_utterance_with_state(event.agent, utterance, state)

    def _has_entity(self, tokens):
        for token in tokens:
            if is_entity(token):
                return True
        return False

    def attach_punct(self, s):
        s = re.sub(r' ([.,!?;])', r'\1', s)
        s = re.sub(r'\.{3,}', r'...', s)
        s = re.sub(r" 's ", r"'s ", s)
        s = re.sub(r" n't ", r"n't ", s)
        return s

    def send(self, is_fake=False):
        tokens, output_data = self.generate(is_fake)

        if is_fake:
            return {'policy': output_data['policies'][1], 'state': output_data['enc_output']}

        if tokens is None:
            return None
        self.dialogue.add_utterance(self.agent, list(tokens))
        # self.dialogue.add_utterance_with_state(self.agent, list(tokens), output_data)

        # if self.agent == 0 :
        #     try:
        #         tokens = [0, 0]
        #         tokens[0] = markers.OFFER
        #         tokens[1] = '$60'
        #     except ValueError:
        #         #return None
        #         pass
        if len(tokens) > 1 and tokens[0] == markers.OFFER and is_entity(tokens[1]):
            try:
                price = self.builder.get_price_number(tokens[1], self.kb)
                return self.offer({'price': price}, metadata=output_data)
            except ValueError:
                #return None
                pass
        tokens = self.builder.entity_to_str(tokens, self.kb)

        if len(tokens) > 0:
            if tokens[0] == markers.ACCEPT:
                return self.accept(metadata=output_data)
            elif tokens[0] == markers.REJECT:
                return self.reject(metadata=output_data)
            elif tokens[0] == markers.QUIT:
                return self.quit(metadata=output_data)

        s = self.attach_punct(' '.join(tokens))
        #print 'send:', s
        return self.message(s, metadata=output_data)

    def step_back(self):
        # Delete utterance from receive
        self.dialogue.delete_last_utterance(delete_state=True)

    def iter_batches(self):
        """Compute the logprob of each generated utterance.
        """
        self.convert_to_int()
        batches = self.batcher.create_batch([self.dialogue])
        yield len(batches)
        for batch in batches:
            # TODO: this should be in batcher
            batch = Batch(batch['encoder_args'],
                          batch['decoder_args'],
                          batch['context_data'],
                          self.env.vocab,
                          num_context=Dialogue.num_context, cuda=self.env.cuda)
            yield batch


class PytorchNeuralSession(NeuralSession):
    def __init__(self, agent, kb, env):
        super(PytorchNeuralSession, self).__init__(agent, kb, env)
        self.vocab = env.vocab

        self.new_turn = False
        self.end_turn = False

    def get_decoder_inputs(self):
        # Don't include EOS
        utterance = self.dialogue._insert_markers(self.agent, [], True)[:-1]
        inputs = self.env.textint_map.text_to_int(utterance, 'decoding')
        inputs = np.array(inputs, dtype=np.int32).reshape([1, -1])
        return inputs

    def _create_batch(self):
        num_context = Dialogue.num_context

        # All turns up to now
        self.convert_to_int()
        encoder_turns = self.batcher._get_turn_batch_at([self.dialogue], Dialogue.ENC, -1)

        encoder_inputs = self.batcher.get_encoder_inputs(encoder_turns)
        encoder_context = self.batcher.get_encoder_context(encoder_turns, num_context)
        encoder_args = {
                        'intent': encoder_inputs[0],
                        'price': encoder_inputs[1],
                        'price_mask': encoder_inputs[2],
                        'context': encoder_context
                    }
        decoder_args = {
                        'intent': None,
                        'price': None,
                        'price_mask': None,
                        'context': self.kb_context_batch,
                    }

        context_data = {
                'agents': [self.agent],
                'kbs': [self.kb],
                }

        return Batch(encoder_args, decoder_args, context_data,
                self.vocab, num_context=num_context, cuda=self.cuda)

    def generate(self, is_fake=False):
        if len(self.dialogue.agents) == 0:
            self.dialogue._add_utterance(1 - self.agent, [], lf={'intent': 'start'})
            # TODO: Need we add an empty state?
        batch = self._create_batch()

        output_data = self.generator.generate_batch(batch, gt_prefix=self.gt_prefix, enc_state=None, whole_policy=is_fake)

        entity_tokens = self._output_to_tokens(output_data)

        return entity_tokens, output_data

    def _is_valid(self, tokens):
        if not tokens:
            return False
        if Vocabulary.UNK in tokens:
            return False
        return True

    def _output_to_tokens(self, data):
        predictions = [data["intent"][0][0], data['price'][0][0]]
        tokens = self.builder.build_target_tokens(predictions, self.kb)
        return tokens

