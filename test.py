from nltk.corpus import wordnet as wn
import math

ALPHA = .2
BETA = .45


# takes two synsets
def similarity(syn1, syn2):
    l_factor = math.exp(-ALPHA * syn1.shortest_path_distance(syn2))

    lch_height = syn1.lowest_common_hypernyms(syn2).min_depth()
    h_power = BETA * lch_height
    pos_h_exp = math.exp(h_power)
    neg_h_exp = 1 / pos_h_exp
    h_factor = (pos_h_exp - neg_h_exp) / (pos_h_exp + neg_h_exp)

    return l_factor * h_factor

class nlpSolver:
    """Hold synsets for clues and candidates"""
    def __init__(clue, crossWord): # takes crossword clue as sentence
        self.clue_synsets = [wn.synsets(word)[0] for word in clue.split(" ")]
        self.candidate_synset = None    # possible solutions to check




def main():

main()
