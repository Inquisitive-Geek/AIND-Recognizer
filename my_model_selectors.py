import math
import statistics
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        component_model_scores = pd.DataFrame({'components_num': [], 'mean_log_likelihood': []})
        for num_states in range(self.min_n_components,self.max_n_components+1):
            hmm_model = self.base_model(num_states)
            BIC = 0
            if hmm_model is not None:
                logL = hmm_model.score(self.X,self.lengths)
                BIC = -2*logL +



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        n_splits = 3
        split_method = KFold(n_splits=n_splits)

        component_model_scores = pd.DataFrame({'components_num': [], 'mean_log_likelihood': []})
        for num_states in range(self.min_n_components,self.max_n_components+1):
            try:
                logL = 0
                count_f = 0
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    # print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx,
                    #                                                           cv_test_idx))  # view indices of the folds
                    try:
                        X_train, X_train_lengths = combine_sequences(cv_train_idx,self.sequences)
                        X_test, X_test_lengths = combine_sequences(cv_test_idx,self.sequences)
                        hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(X_train,
                                                                                                   X_train_lengths)

                        logL += hmm_model.score(X_test, X_test_lengths)
                        # print("model created for {} with {} states".format(self.this_word, num_states))
                    except:
                        count_f += 1
                        print("failure on {} with {} states".format(self.this_word, num_states))
                if count_f < n_splits:
                    logL = logL/(n_splits-count_f)
                component_model_scores.loc[num_states-self.min_n_components] = [num_states,logL]
            except ValueError:
                n_splits = 2
                split_method = KFold(n_splits=n_splits)
                logL = 0
                count_f = 0
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    # print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx,
                    #                                                           cv_test_idx))  # view indices of the folds
                    try:
                        X_train, X_train_lengths = combine_sequences(cv_train_idx,self.sequences)
                        X_test, X_test_lengths = combine_sequences(cv_test_idx,self.sequences)
                        hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(X_train,
                                                                                                   X_train_lengths)

                        logL += hmm_model.score(X_test, X_test_lengths)
                        # print("model created for {} with {} states".format(self.this_word, num_states))
                    except:
                        count_f += 1
                        print("failure on {} with {} states".format(self.this_word, num_states))
                if count_f < n_splits:
                    logL = logL/(n_splits-count_f)
                component_model_scores.loc[num_states-self.min_n_components] = [num_states,logL]
        # If the model build fails, the log likelihood will be 0 in that case
        component_model_scores = component_model_scores[component_model_scores['mean_log_likelihood'] != 0]
        # Best model parameters
        best_num_states = \
            component_model_scores.ix[component_model_scores['mean_log_likelihood'].idxmax()]['components_num']
        print("The best number of states are " + str(best_num_states))
        best_num_states = int(best_num_states)
        try:
            best_hmm_model = GaussianHMM(n_components=best_num_states, covariance_type="diag", n_iter=1000,
                                         random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            # print("model created for {} with {} states".format(self.this_word, best_num_states))
            return best_hmm_model
        except:
            return None


