
import numpy as np
from typing import Union
from collections import defaultdict
import random
from math import e, pi



class GaussianBayes:
    """ Classification by normal law by Bayesian approach
    """
    def __init__(self,):
        self.res = {}
    
    

    def split_data(self, data, weight):
        """
        according to num we split the data and uses the rest rows for testing
        """
        #data_list = list(data)
        #data = np.zeros(data.shape)
        train_size = int(len(data) * weight )
        train_data = []
        for i in range(train_size):
            index = random.randrange(len(data))
            train_data.append(data[index])
            data.pop(index)
        #data = np.asarray(data_list)
        #train_list = np.asarray(train_data)
        return [train_data, data]
    
    def order_by_label(self, data):
        target_map = defaultdict(list)
        for i in range(len(data)):
            features = data [ i ]
            if not features:
                continue
            x = features [-1]
            target_map[x].append(features[:-1])
        return dict(target_map)
        
        
    def prior_prob(self, group, data, target):
        total = float(len(data))
        result = len(group[target]) / total
        return result
    
    def mean(self, num):
        res = sum(num) / float(len(num))
        return res
    
    def compute_standard_deviation(self, num):
        """ compute the standard deviation of numbers """
        moy = self.mean(num)
        err = []
        for i in num:
            diff = (i - moy )**2
            err.append(diff)
        summ = sum(err)
        n =  float(len(num) - 1)
        res = summ / n
        return res ** .5
    
    def score(self, data):
        """
        data_test is a list of features
        this function return the mean and the standard deviation for
        each feature of the train set 
        this is going to be used to calculate the normal probabilty values
        for each feature of test_set
        """
        for label in zip(*data):
            yield {
                    'mean' : self.mean(label),
                    'standard_deviation' : self.compute_standard_deviation(label)
                    }
        
    def priors(self, group, target, data):
        """ this function calculate the prior probabiltiy and returns the probability of each
        class """
        tot = float(len(data))
        res = len(group[target]) / tot
        return res
    
    def train(self, data, target):
        group = self.order_by_label(data)
        self.res = {}
        for target, features in group.items():
            self.res[target] =  {
                    'prior_probability' : self.priors(group, target, data),
                    'score' : [i for i in self.score(features)]
                    }
        return self.res
    
    def fdp_normal(self, x, mean, stdev):
        """
        x variable
        mean u - valeur prédit de samples M
        stdev = var - standard deviation
        retourne : la fonction densité de proabibilité de la gaussienne
        N(x; µ, σ) = (1 / 2πσ) * (e ^ (x–µ)^2/-2σ^2
        """
        var = stdev ** 2
        exp_squared_diff = (x - mean) ** 2
        exp_power = - exp_squared_diff / (2 * var )
        exp = e ** exp_power
        denominator = ((2 * pi) ** .5) * stdev
        prob_normal = exp / denominator
        return prob_normal
        
        
    def joint_probabilities(self, test_list):
        """
        test_list : liste des features ;
        Utilise la fonction fdp_normal(self, x, mean, stdev)
        pour calculer la proabilité en loi normal de chaque label
        Prend le produit de tous les probabilité et les Priors
        """
        probs = {}
        for target, features in self.res.items():
            total_features = len(features['score'])
            likelihood = 1
            for index in range(total_features):
                feature = test_list[index]
                mean = features['score'][index]['mean']
                stdev = features['score'][index]['standard_deviation']
                normal_prob = self.fdp_normal(feature, mean, stdev)
                #print(normal_prob)
                likelihood *= normal_prob
            prior = features['prior_probability']
            #print(prior)
            probs[target] = prior * likelihood
            #print(probs[target])
        return probs
          
    def marginal_fdp(self, joint_probabilities):
        """
        prend en parametre la list des joint probabilities de chaque label 
        et retourne la fonction de densité de proabibilité marginale
        """
        marginal_dist = sum(joint_probabilities.values())
        #print(marginal_dist)
        return marginal_dist
    
    def posterior_probabilities(self, test_list):
        """
        Parametres : test_list ;
        pour chaque label y dans la test_list 
        on calcule : 
            1) Prediction du prior probability
            2) Likelihood 
            3) On multiplie likelihood et priors pour calculer la Loi de probabilité à plusieurs variables
        retourne 
        Prior : P(aa)
        likelihood : P(data[0] | aa) * P(data[1] | aa) 
        Probabilité à plusieurs variables = prior * likelihood
        Prob marginal : prediction prior
        post prob = Probabilité à plusieurs variables / Prob Marginal
        Return  un mapping en dictionnaire de classe en probabilité posterieur 
        """
        posterior_probs = {}
        joint_probabilities = self.joint_probabilities(test_list)
        marginal_prob = self.marginal_fdp(joint_probabilities)
        #print(joint_probabilities)
        #print(marginal_prob)
        for target, joint_prob in joint_probabilities.items():
            posterior_probs[target] = joint_prob / marginal_prob
        return posterior_probs
        
    def get_map(self, test_list):
        """
        Return the target class with the largest/best posterior probability
        Parametre test_list:
        Return target class avec la proabilité posterieur maximale 
        """
        posteriors = self.posterior_probabilities(test_list)
        class_prob = max(posteriors , key=posteriors.get)
        return class_prob
        
    def predict(self, test_set):
        """
        test_set : liste des label 
        return:
            list of predicted targets
        """
        prediction = []
        for row in test_set:
            mapped = self.get_map(row)
            prediction.append(mapped)
        #print(prediction)
        return prediction
        
    def accuracy(self, test_set, predicted):
        """
        ACCURACY OF THE CLASSIFIER
        """
        correct = 0
        actual = [item[-1] for item in test_set]
        for x, y in zip(actual, predicted):
            if x == y:
                correct += 1
        return correct / float(len(test_set))
