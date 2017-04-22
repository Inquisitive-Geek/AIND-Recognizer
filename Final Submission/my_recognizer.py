import warnings
from asl_data import SinglesData
import numpy as np


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    test_set_xlengths = test_set.get_all_Xlengths()
    for word in test_set_xlengths:
        test_set_X, test_set_lengths = test_set_xlengths[word]
        prob_dict = {}
        for model in models:
            try:
                logP = models[model].score(test_set_X,test_set_lengths)
                prob_dict[model] = logP
            except:
                print("The model for word having index " + str(model) + " did not score the word with index " +
                      str(word))
                # In case the model did not return a score, I put in an extremely small number as a proxy
                prob_dict[model] = -1000000000000000000000000000000000000000000000
        probabilities.append(prob_dict)

    for dict in probabilities:
        maxlogL = max(dict.values())
        for key, value in dict.items():
            if value == maxlogL:
                guesses.append(key)

    # for word in test_set:
    #     for model in models:
    #         pass
    return probabilities, guesses

    # raise NotImplementedError
