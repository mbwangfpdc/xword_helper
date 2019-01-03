from nltk.corpus import wordnet as wn
from nltk.corpus import brown as bn
from nltk import FreqDist
import math
import json

with open('data.json', 'r') as browndata:
    data_in = json.load(browndata)

ALPHA = .2
BETA = .45
THRESHOLD = .2
FD = data_in['FD']
BN_LEN = data_in['BN_LEN']
LOG_BN_LEN = data_in['LOG_BN_LEN']

class nlpSolver:
    """Hold synsets for clues and candidates"""
    # takes crossword clue as sentence, crossWord as string
    def __init__(self, clue, crossWord):
        self.clue = clue.split(" ")
        self.clue_synsets = []
        for word in self.clue:
            synsets = wn.synsets(word)
            self.clue_synsets.append(synsets if synsets else None)
        self.syn_idxs = [0] * len(self.clue_synsets)
        self.crossWord = crossWord
        self.candidate_syn = None    # possible solutions to check

    def weight_factor(self, word):
        return 1 - math.log(FD.get(word, 0) + 1) / LOG_BN_LEN

    # takes two synsets
    def sim_calc(self, syn1, syn2):
        #print(syn1)
        #print(syn2)

        syn_dist = syn1.shortest_path_distance(syn2)
        if syn_dist is None:
            return 0
        l_factor = math.exp(-ALPHA * syn_dist)

        lch_height = syn1.lowest_common_hypernyms(syn2)[0].min_depth()
        pos_h_exp = math.exp(BETA * lch_height)
        neg_h_exp = 1 / pos_h_exp
        h_factor = (pos_h_exp - neg_h_exp) / (pos_h_exp + neg_h_exp)

        result = l_factor * h_factor
        return result if result > THRESHOLD else 0

    def sent_sim(self, debug_syn):
        sim_score = 0
        for i in range(0, len(self.clue_synsets)):
            if self.clue_synsets[i] is None:
                continue
            sim_score += self.sim_calc(self.clue_synsets[i][self.syn_idxs[i]], \
                         debug_syn) * self.weight_factor(self.clue[i])
        return sim_score
