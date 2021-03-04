import utils
import time
import math
import numpy as np
import ast
import json
from collections import OrderedDict


class Cafe():
    def __init__(self, X, k=50, s=500, sigma=0.8):
        """
        parametros: 
            - X: conjunto dos termos candidatos
            - k: número de aspectos (default=50)
            - s: indica o número de candidatos que serão agrupados primeiro 
            para gerar os grupos de sementes (default=500)
            - sigma: limite superior da distância entre dois agrupamentos
            (default=0.8)
        """
        self.X = X
        self.k = k
        self.s = s
        self.sigma = sigma
    
    # retorna conjunto de termos candidatos
    def get_X(self):
        return self.X
    
    # retorna o número de aspectos
    def get_k(self):
        return self.k
    
    # retorna o número de candidatos que serão agrupados
    def get_s(self):
        return self.s
    
    # retorna o limite da distância entre dois agrupamentos
    def get_sigma(self):
        return self.sigma
    
    # retorna o conjunto de termos candidatos mais frequentes
    def most_frequent_terms(self, freq_terms):
        
        freq_terms_order = OrderedDict(sorted(freq_terms.items(), key=lambda x: x[1], reverse=True))
        s_num = self.get_s()
        
        candidates_most_freq = set()
        candidates_less_freq = set()
        
        cont = 0
        for k, value in freq_terms_order.items():
            if cont <= s_num:
                candidates_most_freq.add(k)
                cont += 1
            else:
                candidates_less_freq.add(k)
                
        return candidates_most_freq, candidates_less_freq
    
    # retorna lista de conjunto de termos candidatos mais frequentes
    def set_most_frequent_terms(self, freq_terms):
        candidates_most_freq, _ = self.most_frequent_terms(freq_terms)
        
        list_set_candidates = []
        for candidate in candidates_most_freq:
            candi = set()
            cadi.add(candidate)
            list.append(candi)
        
        return list_set_candidates
    
    # retorna lista de conjunto de termos candidatos menos frequentes
    def set_less_frequent_terms(self, freq_terms):
        _, candidates_less_freq = self.most_frequent_terms(freq_terms)
        
        list_set_candidates = []
        for candidate in candidates_less_freq:
            candi = set()
            cadi.add(candidate)
            list.append(candi)
        
        return list_set_candidates
    
    # retorna o um conjunto de dados com todos os clusters mais frequentes
    def set_theta(self, freq_terms):
        candidates_most_freq, _ = self.set_most_frequent_terms(freq_terms)
        return candidates_most_freq
    
    def violate_constraints(cl, cm, sigma):
        return False
    
    # treinamento modelo
    def fit(self, ferq_terms, matrix_terms):
        sigma = self.get_sigma()
        
        theta = self.set_theta(freq_terms)
        list_set_candidates = self.set_most_frequent_terms(freq_terms)
        list_less_candidates = self.set_less_frequent_terms(freq_terms)
        
        idx = 0
        idj = len(list_set_candidates) - 1
        
        set_result = set()
        
        # uni os termos mais frequentes, levando em consideração o Violete Constraints
        # Clusters Sementes
        while len(theta) > 0:
            while idj >= 0: 
                if idx != idj:
                    result = not self.violate_constraints(theta[idx], list_set_candidates[idj], sigma)
                    if result:
                        list_set_candidates[idj] = list_set_candidates[idj].union(theta[idx])
                        _ = theta.pop(idx)
                        idj -= 1
                    else:
                        idj -= 1                       
                else:
                    idj -= 1
                
        # uni os termos menos frequentes, levando em consideração o Violete Constraints
        # Clusters restantes com o mais semelhante entre os sementes
        for index, xi in enumerate(list_less_candidates):
            new_cluster = True
            for idx_i, value in enumerate(list_set_candidates):
                result = not self.violate_constraints(xi, list_set_candidates[idx_i], sigma)
                if result:
                    list_set_candidates[idx_i] = list_set_candidates[idx_i].union(xi)
                    new_cluster = False
                    
            if new_cluster:
                list_set_candidates.appen(xi)
           
        # falta SELECT
        return list_set_candidates
   