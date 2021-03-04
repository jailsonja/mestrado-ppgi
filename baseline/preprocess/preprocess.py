import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk import word_tokenize
from nltk import StanfordTagger
from nltk import StanfordPOSTagger
from sematch.semantic.similarity import WordNetSimilarity, YagoTypeSimilarity
from nltk.corpus import brown, treebank
import pandas as pd
from tqdm import tqdm

exp = ['NN', 'NNS', 'NNP', 'NNPS']
imp = ['JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VM']

wns = WordNetSimilarity()

def cosine_simi(matrix):
    simi = cosine_similarity(matrix)
    return simi

# retorna a similaridade do cosseno entre dois vetores
def cosine(x, y):
    result = np.dot(x, y) / (norm(x) * norm(y))
    return result

def sim_g(x, y):
    return cosine(x, y)

def sim_t(x, y):
    return cosine(x, y)

def sim_gt(x, y):
    result = max(cosine(x, y), cosine(y, x))
    return result

def sim(x, y):
    wg = 0.2
    wt = 0.2
    wgt = 0.6
    value = (wg * simg(x, y)) + (wt * simt(x, y)) + (wgt * simgt(x, y))
    return sim

# calculo PMI
def pmi(freq_x_y, freq_x, freq_y):
    result = np.log(freq_x_y/ freq_x * freq_y)
    return result

# calculo do NPMI
def npmi(freq_x_i, freq_x_j, freq_xi_xj, n):
    if freq_x_i > 0 and freq_x_j > 0 and freq_xi_xj > 0:
        factor1 = (n*freq_xi_xj)/(freq_x_i*freq_x_j)
        factor2 = (freq_xi_xj/n)
        
        result = np.log(factor1)/-np.log(factor2)
        return result
    
    return 0.0

# normalização da matriz de npmi para o intervalo de [0,1]
def normalized_t(matrixt):
    return (matrixt - np.min(matrixt))/np.ptp(matrixt)

def stf_pos_tag(setence):
    jar = '../../stanford-postagger-full-2020-11-17/stanford-postagger.jar'
    model = '../../stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger' # esse mdoelo é melhor que o default do NLTK

    st = StanfordPOSTagger(model, jar, encoding='utf-8')
    words = nltk.word_tokenize(example1)
    tagge_words = st.tag(words)
    
    list_features_exp = []
    list_features_imp = []

    for word, word_class in tagge_words:
        if word_class in exp:
            list_features_exp.append(word)
        
        if word_class in imp:
            list_features_imp.append(word)
        
    return (list_features_exp, list_features_imp)

# retorna a valor da similaridade de duas palavras
def semantic_similarity(w1, w2):
    simi = wns.word_similarity(w1, w2, 'wup')
    return simi

# Função que retorna a distância média em relação a similaridade dos termos dos Clusteres
def dist_avg(clusterl, cluesterm, matrixG, matrixT):
    tam1 = len(clusterl)
    tam2 = len(cluesterm)
    sum_simlarity = 0
    
    for c1 in clusterl:
        for c2 in clusterm:
             sum_simlarity += 1 - sim(matrixG[ci][c2], matrixT[ci][c2])
    return sum_simlarity/(tam1*tam2)

def r_max(cluster, freq_terms):
    v = 0
    r = None
    for key, value in freq_terms.items():
        if value > v:
            v = value
            r = key
    return r

#Função que retorna a distância representativa entre os clusteres em função da
#similaridade dos representantes, os que possuem maior frequência
def dist_rep(clusterl, clusterm, matrixG, matrixT, freq_terms):
    rep1 = r_max(clusterl, freq_terms)
    rep2 = r_max(clusterm, freq_terms)
    
    ans = 1 - sim(matrixG[rep1][rep2], matrixT[rep1][rep2])
    return ans

    
# gera matriz
def generate_matriz(candidates):
    return candidates, candidates
    
# retorna  a matriz de similaridade de termo a termo
def get_matrixG(candidates):
    matrixG = {}
    cd1, cd2 = generate_matriz(candidates)
    for candidate1 in tqdm(cd1, desc='Gerando Matriz de similaridade de termo a termo'):
        matrixG[candidate1] = {}
        for candidate2 in cd2:
            if candidate1 == candidate2:
                matrixG[candidate1][candidate2] = 1               
            matrixG[candidate1][candidate2] = semantic_similarity(candidate1, candidate2)
    return matrixG
            
# retorna matriz de associão estatística de termo a termo
def get_matrixT(candidates, freq_terms, matrix_freq, n):
    cd1, cd2 = generate_matriz(candidates)
            
    matrix = np.ones((len(cd1), len(cd2)))
    
    for idx, candidate1 in tqdm(enumerate(cd1), desc="Gerando Matriz de Associação de Termo a Termo"):
        for idy, candidate2 in enumerate(cd2):
            if candidate1 != candidate2:
                # npmi(freq_x_i, freq_x_j, freq_xi_xj, n):
                matrix[idx, idy] = npmi(freq_terms[candidate1], freq_terms[candidate2], matrix_freq[candidate1][candidate2], n)
                
    matrix_normalized = normalized_t(matrix)
    
    matrixT = {}
    for idx, c1 in tqdm(enumerate(cd1), desc='Normalizando Matriz de Associaçã de Termo a Termo'):
        matrixT[c1] = {}
        for idy, c2 in enumerate(cd2):
            matrixT[c1][c2] = matrix_normalized[idx, idy]
    
    return matrixT