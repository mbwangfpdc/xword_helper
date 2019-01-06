from nltk.corpus import wordnet as wn
from nltk.corpus import brown as bn
# from nltk.corpus import stopwords as sw
from nltk import FreqDist
from copy import deepcopy
import math
import json
import numpy as np
from heapq import heappush, heappushpop, heappop

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
#STOPWORDS = set(sw.words('english'))

DEBUG = False

class ContinueException(Exception):
    pass

class nlpSolver:
    """Hold synsets for clues and candidates"""
    # takes crossword clue as sentence, crossWord as string
    def __init__(self, clue, word_in):
        # self.clue = [w for w in clue.lower().split(" ") if w not in STOPWORDS]
        self.clue = clue.lower().split(" ")
        self.clue_synsets = []
        for word in self.clue:
            synsets = wn.synsets(word)
            self.clue_synsets.append(synsets if synsets else None)
        self.syn_idxs = [0] * len(self.clue_synsets)

        self.word_in = word_in
        self.known_chars = [index for index,char in enumerate(self.word_in) if char != '?']
        # top10 + room for overlaps w/ words in the clue
        self.solutions = [(0, '') for i in range(10 + len(self.clue))]

    def gen_solutions(self):
        for candidate in wn_dict[len(self.word_in)]:
            try:
                for i in self.known_chars:
                    if candidate[i] != self.word_in[i]:
                        raise ContinueException()
            except ContinueException:
                continue
            candidate_syns = wn.synsets(candidate)
            if not candidate_syns:
                continue
            # calc candidate val & poss add to solutions IFF checks passed
            best = max([self.sent_sim(synset) for synset in candidate_syns])

            heappushpop(self.solutions, (best, candidate))
        # end outer for
        # Have 10 solutions
        while self.solutions:
            popt = heappop(self.solutions)
            toprint = (popt[0], popt[1])
            print(toprint)


    def weight_factor(self, word):
        #TODO we ran morphy, so need to run INFLECT for opp. direction
        return 1 - math.log(FD.get(word, 0) + FD.get(wn.morphy(word), 0) + 1) / LOG_BN_LEN

    # takes two synsets
    # FIXME: check for correctness with complete sentence algo
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

    def sent_sim(self, candidate_syn):
        # FIXME: possibly elim stopwords
        # candidate_sent = [w for w in candidate_syn.definition().lower().split(" ") if w not in STOPWORDS]
        candidate_sent = candidate_syn.definition().lower().split(" ")

        seen = set(self.clue)
        all_words = deepcopy(self.clue)
        all_words.extend([x for x in candidate_sent if not (x in seen or seen.add(x))])

        all_synsets = []
        for i,syn in enumerate(self.clue_synsets):
            all_synsets.append(syn[self.syn_idxs[i]])
        for i in range(len(self.clue), len(all_words)):
            # FIXME: itr through all syns of the candidate?
            syn = wn.synsets(all_words[i])
            all_synsets.append(syn[0] if syn else None)

        #FIXME: calc'ing synset 3 times
        candidate_sent_syns = [wn.synsets(w)[0] if wn.synsets(w) else None for w in candidate_sent]

        # get weighted score and order vectors for both the clue sent and candidate sent
        # tuples of index,weighted_score
        try:
            candidate_calc = self.gen_score_vec(candidate_sent, candidate_sent_syns, all_words, all_synsets)
        except IndexError:
            print(candidate_syn)
            exit(0)
        clue_calc = self.gen_score_vec(self.clue, all_synsets[:len(self.clue)], all_words, all_synsets)

        if DEBUG:
            print(candidate_sent)
            print()
            print(all_words)
            print("Cand_calc")
            print(candidate_calc)
            print("Clue_calc")
            print(clue_calc)

            print(FD["ecclesiastical"])
            print(FD["jurisdiction"])

        # END DEBUG

        candidate_calc = (np.array(candidate_calc[0]), np.array(candidate_calc[1]))
        clue_calc = (np.array(clue_calc[0]), np.array(clue_calc[1]))

        return DELTA * ((candidate_calc[1] @ clue_calc[1]) / (np.linalg.norm(candidate_calc[1]) * np.linalg.norm(clue_calc[1]))) \
                     * (1 - DELTA) * np.linalg.norm(clue_calc[0] - candidate_calc[0]) / np.linalg.norm(clue_calc[0] + candidate_calc[0])

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
            try:
                score_vec[i] *= weights[all_words[i]] * weights[sent[best_indices[i] - 1]]
            except IndexError:
                #DEBUG
                print(score_vec)
                print(weights)
                print(all_words)
                print(best_indices)
                print(sent)
                print(i)
                raise IndexError
                #DEBUG
        return (best_indices, score_vec)



    '''
    # old... word -> sentence comparer
    def old_sent_sim(self, candidate_syn):
        sim_score = 0
        for i in range(0, len(self.clue_synsets)):
            if self.clue_synsets[i] is None:
                continue
            sim_score += self.word_sim(self.clue_synsets[i][self.syn_idxs[i]], \
                         candidate_syn) * self.weight_factor(self.clue[i])
        return sim_score
    '''
