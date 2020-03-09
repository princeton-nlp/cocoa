import math
import random
import re
import numpy as np
import torch
from onmt.Utils import use_gpu

from cocoa.model.vocab import Vocabulary
from cocoa.core.entity import is_entity, Entity, CanonicalEntity

from core.event import Event
from .session import Session
from neural.preprocess import markers, Dialogue
from neural.batcher_rl import RLBatch, ToMBatch, RawBatch
from .tom_model import ToMModel
import copy
import time
import json

class NeuralSession(Session):
    def __init__(self, agent, kb, env):
        super(NeuralSession, self).__init__(agent)
        self.env = env
        self.kb = kb
        self.builder = env.utterance_builder
        self.generator = env.dialogue_generator
        self.cuda = env.cuda


        # utterance generator
        self.uttr_gen = env.nlg_module.gen

        self.batcher = self.env.dialogue_batcher
        self.dialogue = Dialogue(agent, kb, None)
        self.dialogue.kb_context_to_int()
        self.kb_context_batch = self.batcher.create_context_batch([self.dialogue], self.batcher.kb_pad)
        self.max_len = 100

        # SL agent type
        # min/expect
        # price strategy: high/low/decay
        self.tom_type = env.tom_type
        self.price_strategy_distribution = {'name': ['insist', 'decay'], 'prob': [0.5, 0.5]}
        self.price_strategy = env.price_strategy
        self.acpt_range = [0.4, 1]

        # Tom
        self.tom = False
        self.controller = None
        if hasattr(env, 'usetom') and env.usetom:
            self.tom = True
            self.critic = env.critic
            self.model = env.model
            self.tom_model = ToMModel(agent, kb, env)
        if env.name == 'pt-neural-r':
            self.sample_price_strategy()

    def sample_price_strategy(self):
        ps = self.price_strategy_distribution['name']
        p = [s for s in self.price_strategy_distribution['prob']]
        self.price_strategy = np.random.choice(ps, p=p)

    @property
    def price_strategy_label(self):
        for i, s in enumerate(self.price_strategy_distribution['name']):
            if s == self.price_strategy:
                return i
        return -1

    def set_controller(self, controller):
        self.controller = controller


    # TODO: move this to preprocess?
    def convert_to_int(self):
        # for i, turn in enumerate(self.dialogue.token_turns):
        #     for curr_turns, stage in zip(self.dialogue.turns, ('encoding', 'decoding', 'target')):
        #         if i >= len(curr_turns):
        #             curr_turns.append(self.env.textint_map.text_to_int(turn, stage))
        self.dialogue.lf_to_int()

    def receive_quit(self):
        e = Event.QuitEvent(self.agent ^ 1, time=self.timestamp())
        self.receive(e)

    # Using another dialogue with semi-event
    def receive(self, event, another_dia=None):
        if isinstance(event, Event) and event.action in Event.decorative_events:
            return
        # print(event.data)
        # Parse utterance
        lf = event.metadata
        utterance = self.env.preprocessor.process_event(event, self.kb)

        # e.g. sentences are here!!
        # when message is offer / accept / reject we do not have "real_uttr"
        # need to be added into state in your ways
        # if "real_uttr" in event.metadata.keys():
        #     print(">>> received sentence", event.metadata["real_uttr"])

        # Empty message
        if lf is None:
            return

        #print 'receive:', utterance
        # self.dialogue.add_utterance(event.agent, utterance)
        # state = event.metadata.copy()
        # state = {'enc_output': event.metadata['enc_output']}

        # utterance_int = self.env.textint_map.text_to_int(utterance)
        # state['action'] = utterance_int[0]
        # print(event.agent, self.dialogue.agent)
        if lf.get('intent') is None:
            print('lf i is None: ', lf)
        if another_dia is None:
            self.dialogue.add_utterance(event.agent, utterance, lf=lf)
        else:
            another_dia.add_utterance(self.dialogue.agent ^ 1, utterance, lf=lf)


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

    def _intent_ind2word(self, ind):
        return self.env.lf_vocab.to_word(ind)

    def _to_event(self, utterance, lf, output_data):
        intent = lf.get('intent')
        intent = self.env.lf_vocab.to_word(intent)
        metadata = {**lf, 'output_data': output_data}
        metadata_nolf = {'output_data': output_data}
        if intent == markers.OFFER:
            return self.offer('offer', metadata)
        elif intent == markers.ACCEPT:
            return self.accept(metadata=metadata)
        elif intent == markers.REJECT:
            return self.reject(metadata=metadata)
        elif intent == markers.QUIT:
            return self.quit(metadata=metadata)
        return self.message(utterance, metadata=metadata)

    def _tokens_to_event(self, tokens, output_data, semi_event=False):
        # if self.agent == 0 :
        #     try:
        #         tokens = [0, 0]
        #         tokens[0] = markers.OFFER
        #         tokens[1] = '$60'
        #     except ValueError:
        #         #return None
        #         pass

        if isinstance(tokens, tuple):
            tokens = list(tokens)

        if isinstance(tokens[1], float):
            tokens[1] = CanonicalEntity(type='price', value=tokens[1])

        if semi_event:
            # From scale to real price
            # print('semi_event: {}->'.format(tokens[1]),end='')
            if tokens[1] is not None:
                tokens[1] = self.builder.get_price_number(tokens[1], self.kb)
            # print('{}.'.format(tokens[1]))
            return tokens

        if isinstance(tokens[0], int):
            tokens[0] = self.env.lf_vocab.to_word(tokens[0])


        if len(tokens) > 1 and tokens[0] == markers.OFFER and is_entity(tokens[1]):
            try:
                price = self.builder.get_price_number(tokens[1], self.kb)
                return self.offer({'price': price}, metadata={"output_data": output_data})
            except ValueError:
                # return None
                pass
        elif tokens[0] == markers.OFFER:
            assert False

        tokens = self.builder.entity_to_str(tokens, self.kb)

        if len(tokens) > 0:
            if tokens[0] == markers.ACCEPT:
                return self.accept(metadata={"output_data": output_data})
            elif tokens[0] == markers.REJECT:
                return self.reject(metadata={"output_data": output_data})
            elif tokens[0] == markers.QUIT:
                return self.quit(metadata={"output_data": output_data})

        while len(tokens) > 0 and tokens[-1] == None: tokens = tokens[:-1]
        s = self.attach_punct(' '.join(tokens))
        # print 'send:', s
        
        # print(">>> sender's intent: ", tokens)
        role = self.kb.facts['personal']['Role']
        category = self.kb.facts['item']['Category']
        real_uttr = self.uttr_gen(tokens, role, category)
        # print(">>> sender's uttr: ", real_uttr)

        return self.message(real_uttr, metadata={"output_data": output_data})

    def get_value(self, all_events):
        all_dia = []
        # print('-'*5+'get_value:')
        #
        # print('in get value:')
        lt0 = last_time = time.time()
        time_list = None
        # for e in all_events:
        #     tmp_tlist = []

            # if info['policy'][act[0]].item() < 1e-7:
            #     continue

            # d = copy.deepcopy(self.dialogue)

            # tmp_tlist.append(time.time() - lt0)
            # lt0 = time.time()

            # self.receive(e, another_dia=d)

            # tmp_tlist.append(time.time() - lt0)
            # lt0 = time.time()

            # d.lf_to_int()

            # tmp_tlist.append(time.time() - lt0)
            # lt0 = time.time()
            # print('='*5)
            # for i, s in enumerate(d.lf_tokens):
            #     print('\t[{}] {}\t{}'.format(d.agents[i], s, d.lfs[i]))

            # all_dia.append(d)

            # tmp_tlist.append(time.time() - lt0)
            # lt0 = time.time()

            # if time_list is None:
            #     time_list = []
            #     for i in range(len(tmp_tlist)):
            #         time_list.append([])
            # for i in range(len(tmp_tlist)):
            #     time_list[i].append(tmp_tlist[i])
        a_r = [self.acc_idx, self.rej_idx]
        qt = [self.quit_idx]
        if all_events[0][0] in a_r:
            # print('a_r', a_r)
            # print('unknown: ', all_events, self.dialogue.token_turns[-1])
            price = None
            for t in self.dialogue.token_turns[-1]:
                if is_entity(t):
                    price = self.builder.get_price_number(t, self.kb)
            values = []
            for e in all_events:
                r = self.controller.get_margin_reward(price=price, agent=self.agent, is_agreed=e[0] == self.acc_idx)
                values.append(r)

            # print('value:', values, price)
            return torch.tensor(values, device=next(self.critic.parameters()).device).view(-1,1)

        if all_events[0][0] in qt:
            is_agreed = ('accept' in self.dialogue.token_turns[-1])
            # print(is_agreed)
            values = [self.controller.get_margin_reward(price=None, agent=self.agent, is_agreed=is_agreed)]
            # print('qt:', all_events, self.dialogue.token_turns[-1])
            # print('value:', values)
            # quit()
            return torch.tensor(values, device=next(self.critic.parameters()).device).view(-1,1)


        attached_events = []
        for e in all_events:
            e = self.env.preprocessor.process_event(e, self.kb)
            attached_events.append({'intent': e[0], 'price': e[1], 'original_price': None})

        # print('copy all dialogue: ', time.time() - last_time)
        # print('for each step: ',end='')
        # for i in range(len(time_list)):
        #     print('{} '.format(np.sum(time_list[i])), end='')
        # last_time = time.time()
        batch = self._create_batch(attached_events=(attached_events, self.dialogue.agent^1))
        # print('create batch: ', time.time() - last_time)
        last_time = time.time()

        # get values
        # batch.mask_last_price()
        e_intent, e_price, e_pmask = batch.encoder_intent, batch.encoder_price, batch.encoder_pmask
        # print('event number:', len(all_events))
        # print('e_intent {}\ne_price{}\ne_pmask{}'.format(e_intent.shape, e_price.shape, e_pmask.shape))
        values = self.critic(e_intent, e_price, e_pmask, batch.encoder_dianum)
        # print('inference: ', time.time() - last_time)
        return values

    def _to_real_price(self, price):
        if price is None: return None
        return self.builder.get_price_number(price, self.kb)

    def _raw_token_to_lf(self, tokens):
        if tokens[-1] is None:
            tokens = tokens[:-1]
        if len(tokens) > 1:
            price = self._to_real_price(tokens[1])
            # print('rt price', tokens[1], type(tokens[1]), price)
            return {'intent': tokens[0], 'price': price}
        return {'intent': tokens[0]}

    def _lf_to_utterance(self, lf):
        role = self.kb.facts['personal']['Role']
        category = self.kb.facts['item']['Category']
        tmplf = lf.copy()
        tmplf['intent'] = self.env.lf_vocab.to_word(tmplf['intent'])
        utterance = self.uttr_gen(tmplf, role, category)
        return utterance

    def tom_inference(self, tokens, output_data):
        # For the step of choosing U2
        # get parameters of normal distribution for price
        p_mean = output_data['price_mean']
        p_logstd = output_data['price_logstd']

        # get all actions
        all_actions = self.generator._get_all_actions(p_mean, p_logstd)
        best_action = (None, None)
        print_list = []

        tom_policy = []
        tom_actions = []

        avg_time = []

        all_value = [-np.inf for i in range(len(all_actions))]

        for act in all_actions:
            if output_data['policy'][0, act[0]].item() < 1e-7:
                continue
            # use fake step to get opponent policy
            tmp_tokens = list(act)
            tmp_lf = self._raw_token_to_lf(act)
            tmp_u = self._lf_to_utterance(tmp_lf)

            self.dialogue.add_utterance(self.agent, tmp_u, lf=tmp_lf)

            # From [0,1] to real price
            e = self._tokens_to_event(tmp_tokens, output_data)
            tmp_time = time.time()
            info = self.controller.fake_step(self.agent, e)
            avg_time.append(time.time() - tmp_time)
            self.dialogue.delete_last_utterance(delete_state=False)
            self.controller.step_back(self.agent)

            tmp = info.exp() * output_data['policy'][0, act[0]]
            # choice the best action
            # if best_action[1] is None or tmp.item() > best_action[1]:
            #     best_action = (tmp_tokens, tmp.item())
            # record all the actions
            tom_policy.append(tmp.item())
            tom_actions.append(tmp_tokens)

            print_list.append((self.env.textint_map.int_to_text([act[0]]), act, tmp.item(), info.item(),
                               output_data['policy'][0, act[0]].item()))

        # print('fake_step costs {} time.'.format(np.mean(avg_time)))

        # Sample action from new policy
        final_action = torch.multinomial(torch.from_numpy(np.array(tom_policy), ), 1).item()
        tokens = list(tom_actions[final_action])

        # print('-'*5+'tom debug info: ', len(self.dialogue.lf_tokens))
        # for s in print_list:
        #     print('\t'+ str(s))
        # self.dialogue.lf_to_int()
        # for s in self.dialogue.lfs:
        #     print(s)

        return tokens

    def try_all_aa(self, tokens, output_data):
        # For the step of choosing U3
        p_mean = output_data['price_mean']
        p_logstd = output_data['price_logstd']
        # get all
        num_price = 5
        all_actions = self.generator._get_all_actions(p_mean, p_logstd, num_price, no_sample=True)
        all_events = []
        new_all_actions = []

        for act in all_actions:
            # TODO: remove continue for min/max tom
            if output_data['policy'][0, act[0]].item() < 1e-7:
                continue
            # Use semi-event here
            #   *For semi-events we only need to transform the price (keep intent as integer)
            # From [0,1] to real price
            e = self._tokens_to_event(act[:2], output_data, semi_event=True)
            # e = self._tokens_to_event(act[:2], output_data)
            all_events.append(e)
            new_all_actions.append(act)
        all_actions = new_all_actions

        print_list = []

        # Get value functions from other one.
        values = self.controller.get_value(self.agent, all_events)

        probs = torch.ones_like(values, device=values.device)
        for i, act in enumerate(all_actions):
            # print('act: ',i ,output_data['policy'], act, probs.shape)
            if act[1] is not None:
                probs[i, 0] = output_data['policy'][0, act[0]].item() * act[2]
            else:
                probs[i, 0] = output_data['policy'][0, act[0]].item()

            print_list.append(
                (self.env.textint_map.int_to_text([act[0]]), act, probs[i, 0].item(), values[i, 0].item()))

        # if len(self.dialogue.lf_tokens) >= 2 and self.dialogue.lf_tokens[-2]['intent'] == 'offer':
        #     print('-' * 5 + 'u3 debug info: ', len(self.dialogue.lf_tokens))
        #     for i, s in enumerate(self.dialogue.lf_tokens):
        #         print('\t[{}] {} {}\t'.format(self.dialogue.agents[i], s, self.dialogue.lfs[i]))
        #     for s in print_list:
        #         print('\t' + str(s))
        # print('is fake: ',time.time()-tmp_time)

        info = {'values': values, 'probs': probs}
        # print('sum of probs', probs.sum())
        # info['values'] = values

        # For the min one
        minone = torch.zeros_like(probs, device=probs.device)
        minone[values.argmin().item(), 0] = 1
        maxone = torch.zeros_like(probs, device=probs.device)
        maxone[values.argmax().item(), 0] = 1

        if self.tom_type == 'expectation':
            # If use expectation here
            return (values.mul(probs)).sum()
        elif self.tom_type == 'competitive':
            # If use max here
            return (values.mul(minone)).sum()
        elif self.tom_type == 'cooperative':
            # If use max here
            return (values.mul(maxone)).sum()
        else:
            print('Unknown tom type: ', self.tom_type)
            assert NotImplementedError()
        return tokens

    def send(self, temperature=1, is_fake=False):

        last_time = time.time()

        acpt_range = None
        if self.env.name == 'pt-neural-r':
            acpt_range = self.acpt_range
        tokens, output_data = self.generate(is_fake=is_fake, temperature=temperature, acpt_range=acpt_range)

        # if self.tom:
        #     print('generate costs {} time.'.format(time.time() - last_time))
        if is_fake:
            tmp_time = time.time()
            tokens = self.try_all_aa(tokens, output_data)

        last_time=time.time()
        if self.tom:
            tokens = self.tom_inference(tokens, output_data)
        # if self.tom:
        #     print('the whole tom staff costs {} times.'.format(time.time() - last_time))

        if tokens is None:
            return None

        lf = self._raw_token_to_lf(tokens)
        utterance = self._lf_to_utterance(lf)

        event = self._to_event(utterance, lf, output_data)
        uttr = self.env.preprocessor.process_event(event, self.kb)
        if uttr is None:
            print('event', event.action, event.metadata)

        price_act = {'price_act': output_data.get('price_act'), 'prob': output_data.get('prob')}
        self.dialogue.add_utterance(self.agent, uttr, lf=lf, price_act=price_act)
        # print('tokens', tokens)

        return event
        # return self._tokens_to_event(tokens, output_data)


    def step_back(self):
        # Delete utterance from receive
        self.dialogue.delete_last_utterance(delete_state=True)

    def iter_batches(self):
        """Compute the logprob of each generated utterance.
        """
        # print('------------dialouge ', self.agent)
        # for i, t in enumerate(self.dialogue.token_turns):
        #     print(self.dialogue.agents[i], t)
        # print('')
        self.convert_to_int()
        batches = self.batcher.create_batch([self.dialogue])
        # print('number of batches: ', len(batches))
        yield len(batches)
        for batch in batches:
            # TODO: this should be in batcher
            batch = RawBatch.generate(batch['encoder_args'],
                          batch['decoder_args'],
                          batch['context_data'],
                          self.env.lf_vocab,
                          cuda=self.env.cuda,)
            yield batch


class PytorchNeuralSession(NeuralSession):
    def __init__(self, agent, kb, env):
        super(PytorchNeuralSession, self).__init__(agent, kb, env)
        self.vocab = env.vocab
        self.lf_vocab = env.lf_vocab
        self.quit_idx = self.lf_vocab.to_ind('quit')
        self.acc_idx = self.lf_vocab.to_ind('accept')
        self.rej_idx = self.lf_vocab.to_ind('reject')

        self.new_turn = False
        self.end_turn = False

    def get_decoder_inputs(self):
        # Don't include EOS
        utterance = self.dialogue._insert_markers(self.agent, [], True)[:-1]
        inputs = self.env.textint_map.text_to_int(utterance, 'decoding')
        inputs = np.array(inputs, dtype=np.int32).reshape([1, -1])
        return inputs

    def _create_batch(self, other_dia=None, attached_events=None):
        num_context = Dialogue.num_context

        # All turns up to now
        self.convert_to_int()
        if other_dia is None:
            dias = [self.dialogue]
        else:
            dias = other_dia

        LF, TOKEN, PACT = Dialogue.LF, Dialogue.TOKEN, Dialogue.PACT
        ROLE = Dialogue.ROLE

        encoder_turns = self.batcher._get_turn_batch_at(dias, LF, -1, step_back=self.batcher.state_length)
        # print('encoder_turns', encoder_turns)
        encoder_tokens = self.batcher._get_turn_batch_at(dias, TOKEN, -1)
        roles = self.batcher._get_turn_batch_at(dias, ROLE, -1)

        encoder_intent, encoder_price, encoder_price_mask = self.batcher.get_encoder_inputs(encoder_turns)

        encoder_args = {
            'intent': encoder_intent,
            'price': encoder_price,
            'price_mask': encoder_price_mask,
            'tokens': encoder_tokens,
        }
        if attached_events is not None:
            i = len(dias[0].lf_turns)+1
        else:
            i = len(dias[0].lf_turns)
        extra = [r + [i / self.batcher.dia_num] + encoder_price[j][-2:] for j, r in enumerate(roles)]
        encoder_args['extra'] = extra
        # encoder_args['dia_num'] = [i / self.dia_num] * len(encoder_intent)

        decoder_args = None

        context_data = {
            'encoder_tokens': encoder_tokens,
            'agents': [self.agent],
            'kbs': [self.kb],
        }
        return RawBatch.generate(encoder_args, decoder_args, context_data,
                self.lf_vocab, cuda=self.cuda)

    def generate(self, temperature=1, is_fake=False, acpt_range=None):
        if len(self.dialogue.agents) == 0:
            self.dialogue._add_utterance(1 - self.agent, [], lf={'intent': 'start'})
            # TODO: Need we add an empty state?
        batch = self._create_batch()
        rlbatch = RLBatch.from_raw(batch, None, None)

        intents, prices = batch.get_pre_info(self.lf_vocab)

        # get acpt_range
        if self.env.name == 'pt-neural-r':
            factor = batch.state[1][0, -3].item()
            if self.price_strategy == 'insist':
                acpt_range = [1., 0.7]
            elif self.price_strategy == 'decay':
                prange = [1., 0.4]
                p = prange[0]*(1-factor) + prange[1]*(factor)
                acpt_range = [1., p-0.1]
            elif self.price_strategy == 'persuaded':
                acpt_range = [1., prices[0, 0].item()]

        output_data = self.generator.generate_batch(rlbatch, enc_state=None, whole_policy=is_fake,
                                                    temperature=temperature, acpt_range=acpt_range)

        # SL Agent with rule-based price action
        if self.env.name == 'pt-neural-r' and self.price_strategy != 'neural' and output_data['price'] is not None:
            # print(output_data['price'])
            oldp = output_data['price']
            if isinstance(oldp, int):
                print('oldp error')
                exit()
            prange = [0, 1]
            prange = acpt_range
            step = 1./5

            # Decay till 1/2 max_length
            factor = batch.state[1][0, -3].item()
            o_factor = factor
            factor = min(1., factor*2)

            if self.price_strategy == 'insist':
                prange = [1., 1.]
                p = prange[0]*(1-factor) + prange[1]*(factor)
            elif self.price_strategy == 'decay':
                prange = [1., 0.4]
                p = prange[0]*(1-factor) + prange[1]*(factor)
                # print('pfactor', p, factor)
            elif self.price_strategy == 'persuaded':
                prange = [1., 0.4]
                if o_factor < 1:
                    # decay
                    p = prices[0, 0].item() - step*(prange[0]-prange[1])
                else:
                    p = prices[0, 0].item()
            else:
                p = oldp
                print('Unknown price strategy: ', self.price_strategy)
                assert NotImplementedError()

            if isinstance(p, int):
                print('p is int')
            # print('prices', prices)
            p = max(p, prices[0, 1].item())
            output_data['price'] = p
            # print('after:', p, output_data['price'])
        if self.env.name != 'pt-neural-r':
            # Using pact
            # 0: insist, 3: decay 0.1, 2: half of last price, 1: agree
            p_act = output_data['original_price']
            p = 1
            if p_act == 0:
                # insist
                p = prices[0, 0].item()
            elif p_act == 1:
                p = prices[0, 1].item()
            elif p_act == 2:
                p = (prices[0, 0].item() + prices[0,1].item())/2
            elif p_act == 3:
                p = prices[0, 0].item()-0.1
            else:
                print('what\'s wrong?', p_act)
            p = max(p, prices[0, 1].item())

            output_data['original_price'] = p
            if output_data['price'] is not None:
                output_data['price'] = p
            output_data['price_act'] = p_act


        if isinstance(output_data['price'], int):
            print('op is int', prices.dtype, self.env.name, self.price_strategy)

        entity_tokens = self._output_to_tokens(output_data)

        return entity_tokens, output_data

    def _is_valid(self, tokens):
        if not tokens:
            return False
        if Vocabulary.UNK in tokens:
            return False
        return True

    def _output_to_tokens(self, data):
        # print(data['intent'], data['price'])
        if isinstance(data['intent'], int):
            predictions = [data["intent"]]
        elif isinstance(data['intent'], torch.Tensor):
            predictions = [data["intent"].item()]

        if data.get('price') is not None:
            p = data['price']
            p = max(p, 0.0)
            p = min(p, 1.0)
            predictions += [p]
        else:
            predictions += [None]

        # tokens = self.builder.build_target_tokens(predictions, self.kb)
        # print('out_to_tokens', predictions, tokens)
        # print('converting to tokens: {} -> {}'.format(predictions, tokens))
        return predictions

