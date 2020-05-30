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
import copy
import time
import json

class ToMModel(object):

    def __init__(self, agent, kb, env):
        self.agent = agent
        self.env = env
        self.kb = kb
        self.generator = env.tom_generator
        self.gt_session = None
        self.dialogue = Dialogue(agent, kb, None)
        self.dialogue.kb_context_to_int()
        self.hidden_state = None

    def estimate(self, uttr, lf):
        self.update_dialogue(uttr, lf)
        # ...
        # for all a1



        self.recover_dialogue()

    def update_dialogue(self, uttr, lf):
        self.dialogue.add_utterance(1-self.agent, uttr, lf=lf)

    def recover_dialogue(self):
        self.dialogue.delete_last_utterance()