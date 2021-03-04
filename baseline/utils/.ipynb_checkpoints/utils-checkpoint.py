import pandas as pd
import numpy as np
import json
import time
from tqdm import tqdm

def read_features_txt(product):
    file_path_exp = './database/gold_standard/explicit_features_{}.txt'.format(product)
    file_path_imp = './database/gold_standard/implicit_features_{}.txt'.format(product)
    
    list_feature = []
    f =  open(file_path_exp, 'r')
    for line in f:
        term = line.split('\n')[0]
        list_feature.append(term)
    f.close()
    
    f =  open(file_path_imp, 'r')
    for line in f:
        term = line.split('\n')[0]
        list_feature.append(term)
    f.close()
    
    return list_feature

def read_reviews_txt(product):
    file_path_rew = './database/reviews_{}.txt'.format(product)
    
    documents = {}
    
    #lendo arquivo de reviews
    f = open(file_path_rew, 'r', encoding = "ISO-8859-1")
    rws = None
    idx = 0
    for line in f:
        if 'review_text: ' in line:
            rws = line[13:].split('\n')[0]
            documents[idx] = rws
            idx += 1
    f.close()
    
    return documents
                        
def set_candidate_terms(product):
    # ler aspectos explicitos e implicitos das sentencas de um produto
    candidates = read_features_txt(product)
    candidates = set(candidates)
    return candidates

def reviews_frequencies(product):
    dict_terms_freq = {}
    matrix_freq = {}
    
    #lendo arquivo de reviews
    documents = read_reviews_txt(product)
    
    # get termos do produto
    termos = set_candidate_terms(product)
        
    # frequencia termo documentos
    for termo in tqdm(termos, desc="Frequencia termos documento"):
        cont = 0
        for _, value in documents.items():
            if termo in value:
                cont += 1
        dict_terms_freq[termo] = cont
            
    # frequencia termos documentos
    for tm1 in tqdm(termos, desc="Frequencia 2 termos em um mesmo documentos"):
        matrix_freq[tm1] = {}
        for tm2 in termos:
            cont = 0
            for _, value in documents.items():
                if tm1 == tm2:
                    matrix_freq[tm1][tm2] = 0
                if tm1 != tm2:
                    if tm1 in value:
                        if tm2 in value:
                            cont += 1
            matrix_freq[tm1][tm2] = cont
                
    return dict_terms_freq, matrix_freq

# retorna o número de documentos de um produto
def get_numbers_documents(product):
    documents = read_reviews_txt(product)
    return len(documents)