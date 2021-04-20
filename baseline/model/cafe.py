from preprocess.preprocess import dist_rep, dist_avg, stf_word
import time
import math
import numpy as np
import ast
import json
from collections import OrderedDict
from tqdm import tqdm
import time

exp = ['NN', 'NNS', 'NNP', 'NNPS']
imp = ['JJ', 'JJR', 'JJS', 'VB']
class Cafe():
    def __init__(self, X, X_tag,k=50, s=500, sigma=0.8):
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
        self.X_tag = X_tag
        self.k = k
        self.s = s
        self.sigma = sigma
    
    # retorna conjunto de termos candidatos
    def get_X(self):
        return self.X
    
    def get_X_tag(self):
        return self.X_tag
    
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
            
            list_set_candidates.append([candidate])
        
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
        candidates_less_freq = list(candidates_less_freq)
        
        return candidates_most_freq, candidates_less_freq
    
    def list_tuple(self, setences):
        set_tuple = set()
        for setence in setences:
            set_tuple.add((setence[0], setence[1]))
        return set_tuple
    
    # função responsável por checar a métrica 3, descrita no artigo
    def check_metric3(self, cl, cm, documents_setences):
        x = 0
        y = 0
        for l in cl:
            for m in cm:
                d1 = self.list_tuple(documents_setences[l])
                d2 = self.list_tuple(documents_setences[m])
                if(len(d1) > 0 and len(d2) > 0):
                    inter = d1.intersection(d2)
                    union = d1.union(d2)
                    x += len(inter)
                    
                    if len(inter) > 0:
                        resto = union - inter
                        for r in resto:
                            if r[1] == list(inter)[0][1]:
                                y += 1
        result = x > y
        return result
    
    # funçao responsável por checar a métrica 2, descrita no artigo
    def check_metric2(self, cl, cm):
        candidates_tag = self.X_tag
        #value = False
        
        for i in cl:
            target = candidates_tag[i]
            if target in exp:
                value = True
                break
            else:
                value = False
        
        if not value:
            for i in cm:
                target = candidates_tag[i]
                if target in exp:
                    value = True
                    break
        return value
    
    # função responsável por verificar se os clusters podem ser unidos
    def violate_constraints(self, cl, cm, matrixG, matrixT, freq_terms, documents_setences):
        dm = dist_avg(cl, cm, matrixG, matrixT) #Distancia média entre os clusteres
        dr = dist_rep(cl, cm, matrixG, matrixT, freq_terms) #Distancia entre os representantes dos clusteres
        
        distancia = max(dm, dr) #Distancia maxima
        #print('SIGMA: ', self.sigma)
        
        metrica1 = (distancia <= self.sigma) #Verifica se viola a regra de distancia V
        metrica2 = self.check_metric2(cl, cm) # V
        metrica3 = self.check_metric3(cl, cm, documents_setences) # V
        
        print('DM: ', dm, 'DR: ', dr, 'DST: ', distancia)
        print(metrica1, metrica2, metrica2)
        print()
        return not (metrica1 and metrica2 and metrica3)
    
    def mergeable_clusters(self, cl, cm, matrixG, matrixT, freq_terms, documents_setences):
        result = self.violate_constraints(cl, cm, matrixG, matrixT, freq_terms, documents_setences)
        if result:
            return False
        else:
            return True
    # Agrupa os termos menos frequentes do conjunto de termos candidatos
    def merge_less_candidates(self, element, theta, matrixG, matrixT, freq_terms, documents_setences):
        new = True
        for idx, value in enumerate(theta):
            #termo = set()
            #termo.add(candidate)
            result = self.mergeable_clusters(theta[idx], [element], matrixG, matrixT, freq_terms, documents_setences)
            if result:
                print(theta[idx], element)
                theta[idx].extend([element])
                new = False
                break
        if new:
            theta.append([element])
            
        return theta
        
    # treinamento modelo
    def discovery_cluster(self, freq_terms, matrixG, matrixT, documents_setences):
        sigma = self.get_sigma()
        
        theta, X_1 = self.most_frequent_terms(freq_terms) 
        # Clusters Sementes

        i = 0
        while i < len(theta):
            j = 0
            while j < len(theta):                 
                if i != j:                        
                    print(i, j , len(theta))
                        
                    result = self.mergeable_clusters(theta[i], theta[j], matrixG, matrixT, freq_terms,documents_setences)
                    if result:
                        #if(j > i):
                        #    theta[i].extend(theta[j])
                        #    theta.remove(theta[j])
                        #else:
                        theta[j].extend(theta[i])
                        theta.remove(theta[i])
                    else:
                        j += 1
                else:
                    j += 1
                    
            i += 1

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
        elif len(groups) > self.get_k():
            cont = 0
            for key, value in tqdm(groups.items(), desc='Selecionando K clusters'):
                if cont <= self.get_k():
                    v = value[1]
                    results.append(v)
                    cont += 1
                else:
                    break
        
            return results
   