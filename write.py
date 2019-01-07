from nltk.corpus import wordnet as wn
from nltk.corpus import brown as bn
from nltk.corpus import words
from nltk import FreqDist
import math
import json
'''
FD = FreqDist([w.lower() for w in bn.words() if w.isalpha()])
BN_LEN = len(bn.words())
LOG_BN_LEN = math.log(BN_LEN + 1)
datum = {'FD':FD, 'BN_LEN':BN_LEN, 'LOG_BN_LEN':LOG_BN_LEN}
with open('data.json', 'w+') as datafile:
    json.dump(datum, datafile)
'''

all_lemmas = list(wn.all_lemma_names())

print(len(all_lemmas))

non_alpha_chars = {}

GOOD = ["'","_","-","."]
BAD = ['2', '3', '8', '4', '5', '0', '1', '6', '7', '9', '/']

woords = [[] for i in range(65)]

# TODO: woords make into dictionary
for lemma in all_lemmas:
    if any(char in lemma for char in BAD):
        continue
    elif lemma.isalpha():
        woords[len(lemma)].append((lemma, None))
    else:
        stripped_lemma = lemma
        for char in GOOD:
            stripped_lemma = stripped_lemma.replace(char, "")
        if not stripped_lemma.isalpha():
            raise ContinueException
        try:
            woords[len(stripped_lemma)].append((lemma, stripped_lemma))
        except IndexError as e:
            print(e)
            print(stripped_lemma)

with open('dictionary.json', 'w+') as dictfile:
    json.dump(woords, dictfile)
