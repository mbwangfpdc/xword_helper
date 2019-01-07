from nltk.corpus import wordnet as wn
from nlpSolver import *
import unittest

# test some synsets for reasonable scores
a = wn.synsets('rock')[0]
#b = wn.synsets('memory')[3]


class TestNLPSolver(unittest.TestCase):
    def __init__(self):
        self.solver = nlpSolver("ecclesiastical jurisdiction", "s????")
        self.solver.gen_solutions()
        # print(self.solver.sent_sim(a))
    def test_close(self):
        pass


    def test_mid(self):
        pass


    def test_far(self):
        pass


mysolver = TestNLPSolver()
