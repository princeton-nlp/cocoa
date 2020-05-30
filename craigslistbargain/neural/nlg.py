## NLG module

import json
import difflib
# from termcolor import colored
import string
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
import numpy as np
import random
from cocoa.core.entity import is_entity, Entity, CanonicalEntity

class IRNLG(object):
    def __init__(self, args):
        self.gen_dic = {}
        with open(args.nlg_dir) as json_file:  
            self.gen_dic = json.load(json_file)
        # self.gen_dic = {}
        # for tmp in self.templates:
        #     if not tmp["category"] in self.gen_dic.keys():
        #         self.gen_dic[tmp["category"]] = {}
        #     if not tmp["intent"] in self.gen_dic[tmp["category"]].keys():
        #         self.gen_dic[tmp["category"]][tmp["intent"]] = {}
        #     if tmp["role"] in self.gen_dic[tmp["category"]][tmp["intent"]].keys():
        #         self.gen_dic[tmp["category"]][tmp["intent"]][tmp["role"]].append(tmp["template"])
        #     else:
        #         self.gen_dic[tmp["category"]][tmp["intent"]][tmp["role"]] = [tmp["template"]]

    def gen(self, lf, role, category, as_tokens=False):
        if self.gen_dic[category].get(lf.get('intent')) is None:
            # print('not in nlg:', lf, role, category)
            return [''], (lf.get('intent'), role, category, 0)
        tid = random.randint(0, len(self.gen_dic[category][lf.get('intent')][role])-1)
        template = self.gen_dic[category][lf.get('intent')][role][tid]
        words = word_tokenize(template)
        new_words = []
        for i, wd in enumerate(words):
            if wd == "PPRRIICCEE" and lf.get('price'):
                if as_tokens:
                    new_words.append(CanonicalEntity(type='price', value=lf.get('price')))
                else:
                    new_words.append('$'+str(lf.get('price')))
            else:
                new_words.append(wd)

        # TODO: raw uttrence
        if as_tokens:
            return new_words, (lf.get('intent'), role, category, tid)

        sentence = "".join([" "+i if not i.startswith("'") and i not in string.punctuation
                        else i for i in new_words]).strip()

        return sentence, (lf.get('intent'), role, category, tid)