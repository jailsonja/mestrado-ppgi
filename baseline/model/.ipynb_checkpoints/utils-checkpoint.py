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

exp = ['NN', 'NNS', 'NNP', 'NNPS']
imp = ['JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VM']

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
    return np.log(n*(freq_xi_xj)/freq_x_i * freq_x_j) / -np.log(freq_xi_xj/n)

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

def semantic_similarity(w1, w2):
    wns = WordNetSimilarity()
    simi = wns.word_similarity(w1, w2, 'wup')
    return simi