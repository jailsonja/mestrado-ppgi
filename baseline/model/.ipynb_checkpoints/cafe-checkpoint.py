from preprocess.preprocess import dist_rep, dist_avg, stf_word
import time
import math
import numpy as np
import ast
import json
from collections import OrderedDict
from utils.utils import read_documents_setences
from tqdm import tqdm
import time

class Cafe():
    def __init__(self, X, k=50, s=50, sigma=0.8):
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
    
    def set_to_list_set(self, set_candidates):
        list_set_candidates = []
        for candidate in set_candidates:
            candi = set()
            candi.add(candidate)
            list_set_candidates.append(candi)
        
        return list_set_candidates
    
    # retorna o conjunto de termos candidatos mais frequentes e menos frequentes
    def most_frequent_terms(self, freq_terms):
        
        freq_terms_order =dict(sorted(freq_terms.items(), key=lambda item: item[1], reverse=True))
        s_num = self.get_s()
        
        candidates_most_freq = set()
        candidates_less_freq = set()
        
        cont = 0
        for k, value in freq_terms_order.items():
            if cont < s_num:
                candidates_most_freq.add(k)
                cont += 1
            else:
                candidates_less_freq.add(k)
        
        candidates_most_freq = self.set_to_list_set(candidates_most_freq)
        return (candidates_most_freq, candidates_less_freq)
    
    # função responsável por checar a métrica 3, descrita no artigo
    def check_metric3(self, cl, cm, documents_setences):
        x = 0
        y = 0        
        for l in cl:
            for m in cm:
                for key, valor in documents_setences.items():
                    d1 = set(valor[l])
                    d2 = set(valor[m])
                    if(len(d1) > 0 and len(d2) > 0):
                        inter = d1.intersection(d2)
                        union = d1.union(d2)
                        x += len(inter)
                        y += len(union - inter)
                        
        return x > y
    # funç~çao responsável por checar a métrica 2, descrita no artigo
    def check_metric2(self, cl, cm):
        for i in cl:
            target = stf_word(i)
            if target:
                break
                
        if target:
            return target
        
        for i in cm:
            target = stf_word(i)
            if target:
                break
        return target
    
    # função responsável por verificar se os clusters podem ser unidos
    def violate_constraints(self, cl, cm, matrixG, matrixT, freq_terms, documents_setences):
        dm = dist_avg(cl, cm, matrixG, matrixT) #Distancia média entre os clusteres
        dr = dist_rep(cl, cm, matrixG, matrixT, freq_terms) #Distancia entre os representantes dos clusteres
        
        distancia = max(dm, dr) #Distancia maxima
        
        metrica1 = (distancia <= self.get_sigma()) #Verifica se viola a regra de distancia
        metrica2 = self.check_metric2(cl, cm)
        metrica3 = self.check_metric3(cl, cm, documents_setences)
        
        return metrica1 and metrica2 and metrica3
    
    # Agrupa os termos menos frequentes do conjunto de termos candidatos
    def merge_less_candidates(self, candidate, theta, matrixG, matrixT, freq_terms, documents_setences):
        new = True
        for idx, value in enumerate(theta):
            termo = set()
            termo.add(candidate)
            result = self.violate_constraints(theta[idx], termo, matrixG, matrixT, freq_terms, documents_setences)
            if result:
                theta[idx] = theta[idx].union(termo)
                new = False
                break                
        if new:
            termo = set()
            termo.add(candidate)
            theta.append(termo)
            
        return theta
        
    # treinamento modelo
    def discovery_cluster(self, freq_terms, matrixG, matrixT, documents_setences):
        sigma = self.get_sigma()
        
        X = self.most_frequent_terms(freq_terms) # todos os termos candidatos
        
        theta = X[0] # termos mais frequentes        
        X_1 = X[1] # termos menos frequentes
        
        idx = 0

        set_result = set()
        
        # uni os termos mais frequentes, levando em consideração o Violete Constraints
        # Clusters Sementes
        pbar = tqdm(total=len(theta))
        
        while idx < len(theta):
            idj = 0
            while idj < len(theta):
                if idx != idj:
                    result = self.violate_constraints(theta[idx], theta[idj], matrixG, matrixT, freq_terms,documents_setences)
                    if result:
                        if idj > idx:
                            print('IDX: ', idj, 'IDJ: ', idx)
                            print(theta[idj], theta[idx])
                            theta[idx] = theta[idx].union(theta[idj])
                            theta.remove(theta[idj])
                        else:
                            print('IDX: ', idx, 'IDJ: ', idj)
                            print(theta[idj], theta[idx])
                            theta[idj] = theta[idj].union(theta[idx])
                            theta.remove(theta[idx])
                    else:
                        idj += 1
                        pbar.update(1)
                else:
                    idj += 1
                    pbar.update(1)
                    
            idx += 1
            pbar.update(1)
            
        pbar.close()
                
        # uni os termos menos frequentes, levando em consideração o Violete Constraints
        # Clusters restantes com o mais semelhante entre os sementes
        
        for x1 in tqdm(X_1, desc="Mapeando Termos Menos Frequente"):
            theta = self.merge_less_candidates(x1, theta, matrixG, matrixT, freq_terms, documents_setences)
       
        return theta
        
    # Seleciona os k cluters resultantes do agrupamento
    def select(self, freq_terms, cluster_results):
        groups = {}
        results = []
        for idx, cluster in enumerate(cluster_results):
            soma = 0
            lista_clu = []
            for clu in cluster:
                soma += freq_terms[clu]
                lista_clu.append(clu)
            value = (soma, lista_clu)
            groups[idx] = value
            
        groups = dict(sorted(groups.items(),key=lambda x: x[1][0], reverse=True))
        
        if len(groups) < self.get_k():
            for key, value in tqdm(groups.items(), desc='Selecionando K clusters'):
                v = value[1]
                results.append(v)
            return results
        elif len(groups) >= self.get_k():
            cont = 0
            for key, value in tqdm(groups.items(), desc='Selecionando K clusters'):
                if cont <= self.get_k():
                    v = value[1]
                    results.append(v)
                    cont += 1
                break
            return results
   