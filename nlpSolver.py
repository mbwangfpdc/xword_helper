from nltk.corpus import wordnet as wn
from nltk.corpus import brown as bn
from nltk.corpus import stopwords as sw
from nltk.stem.snowball import SnowballStemmer
from nltk import FreqDist
from copy import deepcopy
import math
import json
import numpy as np
from heapq import heappush, heappushpop, heappop
import re

XWORD_PATTERN = re.compile("[^a-zA-Z\-_' ]")

import pprint

with open('data.json', 'r') as browndata:
    data_in = json.load(browndata)
with open('dictionary.json', 'r') as wn_dict:
    wn_dict = json.load(wn_dict)

ALPHA = .2
BETA = .45
DELTA = .85
THRESHOLD = .2
FD = data_in['FD']
BN_LEN = data_in['BN_LEN']
LOG_BN_LEN = data_in['LOG_BN_LEN']
STOPWORDS = set(sw.words('english'))
stemmer = SnowballStemmer("english")

#TODO: utilize this to eliminate nytimes crossword operator words
XWORD_OPS = {'e.g.'}

DEBUG_FLAG = '0'
DEBUG_CHECK_WORD_SCORE = True

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def smart_stem(word):
    stem = stemmer.stem(word)
    return stem if FD.get(stem, 0) > FD.get(word, 0) else word

# cleans string input -> synsets for clues, candidates
def gen_synsets(sentence):
    synsets = [wn.synsets(w)[0] if wn.synsets(w) else None for w in sentence]
    #lower
    #regex simplify
    #None if is STOPWORDS
    #None if no synsets


class ContinueException(Exception):
    pass

DEBUG_STATE = {
    "__CANDIDATE__": {
        "all": ["all", "words"],
        "clue_score": [0.7, 0.5],
        "cand_score": [0.5, 0.7],
        "clue_order": [1, 0],
        "cand_order": [0, 1],
    },
}

class nlpSolver:
    """Hold synsets for clues and candidates"""
    # takes crossword clue as sentence, crossWord as string
    def __init__(self, clue, word_in):
        # self.clue = [w for w in clue.lower().split(" ") if w not in STOPWORDS]
        # TODO prolly need to Morphy() all words in clue
        self.clue = clue.lower().split(" ")
        self.orig_clue = self.clue
        self.clue = [smart_stem(w) for w in self.clue]
        self.clue_synsets = []
        for word in self.clue:
            synsets = wn.synsets(word)
            self.clue_synsets.append(synsets[0] if synsets else None)
        # print(self.clue_synsets
        self.word_in = word_in.lower()
        self.known_chars = [index for index,char in enumerate(self.word_in) if char != '?']
        # top10 + room for overlaps w/ words in the clue
        self.solutions = [(0, '') for i in range(10 + len(self.clue))]
        self.seek_guidance(self.clue, self.clue_synsets)
        if len(self.clue) == 1:
            self.clue = XWORD_PATTERN.sub("", self.clue_synsets[0].definition()).lower().split(" ")
            self.clue_synsets = [wn.synsets(w)[0] if wn.synsets(w) else None for w in self.clue]
            #FIXME seeking too much guidance
            self.seek_guidance(self.clue, self.clue_synsets)

        self.current_candidate = None

    def seek_guidance(self, sentence, sent_synsets):
        while(True):
            for idx, (w_syn) in enumerate(zip(sentence, sent_synsets)):
                print("{}: {}... {}".format(idx, w_syn[0], w_syn[1].definition() if w_syn[1] else ""))

            try:
                word_idx = int(input("\nEnter # of word for which you wish to provide guidance. ('q' to quit)\n"))
            except ValueError:
                return

            for idx, syn in enumerate(wn.synsets(sentence[word_idx])):
                print("{}: {}".format(idx, syn.definition()))

            syn_idx = int(input("\nSelect # of correct redefinition from above\n"))

            sent_synsets[word_idx] = wn.synsets(sentence[word_idx])[syn_idx]


    def gen_solutions(self):
        feelsgoodman = 0 #DEBUG

        # (REAL, STRIPPED) e.g. ("mr.big_butt", "mrbigbutt")
        for candidate in wn_dict[len(self.word_in)]:
            candidate_raw = candidate[0]
            self.current_candidate = candidate_raw
            candidate_stripped = candidate[1] or candidate[0]
            try:
                for i in self.known_chars:
                    if candidate_stripped[i] != self.word_in[i]:
                        raise ContinueException()
            except ContinueException:
                continue
            candidate_syns = wn.synsets(candidate_raw)
            if not candidate_syns:
                continue
            # calc candidate val & poss add to solutions IFF checks passed
            best = max([self.sent_sim(synset) for synset in candidate_syns])

            feelsgoodman += 1 # feelsgood DEBUG OH YEAH
            heappushpop(self.solutions, (best, candidate_raw))
        # end outer for

        # Have 10 solutions
        while self.solutions:
            score, solution = heappop(self.solutions)
            toprint = (score, solution)
            print(toprint)
            '''
            if DEBUG_FLAG is 'a':
                # re-find correct synset out of synsets for debugging
                syn = max(wn.synsets(solution), key=lambda synset: self.sent_sim(synset))
                soln_def = XWORD_PATTERN.sub("", syn.definition()).lower().split(" ")
                soln_def = [smart_stem(w) for w in soln_def]
                all_words, all_syns = self.gen_all_vecs(soln_def)
                r1, s1 = self.gen_score_vec(soln_def, [(wn.synsets(w)[0] if wn.synsets(w) else None) for w in soln_def], all_words, all_syns)
                r2, s2 = self.gen_score_vec(self.clue, self.clue_synsets, all_words, all_syns)
                s1 = [truncate(fl, 3) for fl in s1]
                s2 = [truncate(fl, 3) for fl in s2]
                for l in [s1, s2, r1, r2]:
                    for score in l:
                        print('{:5}\n'.format(score), end=' ')
                for w in all_words:
                    print('{:5}\n)'.format(w[:5]), end=' ')
                print(all_words)
                print()
            '''

        if DEBUG_CHECK_WORD_SCORE:
            print("How good does it feel?\n.\n.\n.\n.\n{} feelsgoodmen\n".format(feelsgoodman))

    def weight_factor(self, word):
        #TODO we ran morphy, so need to run INFLECT for opp. direction
        return 1 - math.log(FD.get(word, 0) + 1) / LOG_BN_LEN

    # takes two synsets
    # FIXME: check for correctness with complete sentence alg
    def word_sim(self, syn1, syn2):
        if syn1 is None or syn2 is None:
            return 0

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

    def gen_all_vecs(self, candidate_sent):
        seen = set(self.clue)
        all_words = deepcopy(self.clue)
        all_words.extend([x for x in candidate_sent if not (x in seen or seen.add(x))])

        all_synsets = []
        all_synsets.extend(self.clue_synsets)
        for i in range(len(self.clue), len(all_words)):
            # FIXME: itr through all syns of the candidate?
            syn = wn.synsets(all_words[i])
            all_synsets.append(syn[0] if syn else None)

        if DEBUG_FLAG == 'b':
            DEBUG_STATE[self.current_candidate] = {}
            DEBUG_STATE[self.current_candidate]["all"] = zip(all_words, [syn.name() for syn in all_synsets])

        return (all_words, all_synsets)

    def semantic_sim(self, score_vec1, score_vec2):
        return DELTA * ((score_vec1 @ score_vec2) / (np.linalg.norm(score_vec1) * np.linalg.norm(score_vec2)))

    def syntactic_sim(self, order_vec1, order_vec2):
        return (1 - DELTA) * np.linalg.norm(order_vec1 - order_vec2) / np.linalg.norm(order_vec1 + order_vec2)

    def sent_sim(self, candidate_syn):
        # FIXME: possibly elim stopwords
        # candidate_sent = [w for w in candidate_syn.definition().lower().split(" ") if w not in STOPWORDS]
        candidate_sent = XWORD_PATTERN.sub("", candidate_syn.definition()).lower().split(" ")
        candidate_sent = [smart_stem(w) for w in candidate_sent]

        all_words, all_synsets = self.gen_all_vecs(candidate_sent)

        #FIXME: calc'ing synset 3 times
        candidate_sent_syns = [wn.synsets(w)[0] if wn.synsets(w) else None for w in candidate_sent]

        # get weighted score and order vectors for both the clue sent and candidate sent
        # tuples of index,weighted_score
        candidate_calc = self.gen_score_vec(candidate_sent, candidate_sent_syns, all_words, all_synsets)
        clue_calc = self.gen_score_vec(self.clue, self.clue_synsets, all_words, all_synsets)

        candidate_calc = (np.array(candidate_calc[0]), np.array(candidate_calc[1]))
        clue_calc = (np.array(clue_calc[0]), np.array(clue_calc[1]))

        if DEBUG_FLAG == 'b':
            DEBUG_STATE[self.current_candidate]["clue_score"] = deepcopy(clue_calc[0])
            DEBUG_STATE[self.current_candidate]["cand_score"] = deepcopy(candidate_calc[0])
            DEBUG_STATE[self.current_candidate]["clue_order"] = deepcopy(clue_calc[1])
            DEBUG_STATE[self.current_candidate]["cand_order"] = deepcopy(candidate_calc[1])

        return self.syntactic_sim(candidate_calc[0], clue_calc[0]) + self.semantic_sim(candidate_calc[1], clue_calc[1])

    def gen_score_vec(self, sent, sent_syns, all_words, all_syns):
        best_indices = [0] * len(all_words)
        score_vec = [0.0] * len(all_words)

        for i in range(len(all_words)):
            try:
                # if w1 in sent already
                best_indices[i] = sent.index(all_words[i]) + 1
                score_vec[i] = 1.0
            except ValueError:
                # else do calc
                # for j,syn in enumerate(sent_syns):
                best = (0, 0.0)
                for j in range(len(sent)):
                    score = self.word_sim(sent_syns[j], all_syns[i])
                    if score > best[1]:
                        best = (j, score)
                best_indices[i] = best[0]
                score_vec[i] = best[1]

        # FIXME calc'ing weight many times definitely
        # but using this dict for now, maybe move to outer code
        weights = {}
        for word in all_words:
            weights[word] = self.weight_factor(word)

        # after calculating score, weight 'em
        for i in range(len(score_vec)):
            score_vec[i] *= weights[all_words[i]] * weights[sent[best_indices[i] - 1]]
        return (best_indices, score_vec)
