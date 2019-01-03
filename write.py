from nltk.corpus import wordnet as wn
from nltk.corpus import brown as bn
from nltk import FreqDist
import math
import json

FD = FreqDist([w.lower() for w in bn.words() if w.isalpha()])
BN_LEN = len(bn.words())
LOG_BN_LEN = math.log(BN_LEN + 1)
datum = {'FD':FD, 'BN_LEN':BN_LEN, 'LOG_BN_LEN':LOG_BN_LEN}
with open('data.json', 'w+') as datafile:
    json.dump(datum, datafile)
