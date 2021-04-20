import pandas as pd
import numpy as np

import json
import time
from tqdm import tqdm
import string

from preprocess.preprocess import stf_pos_tag

import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize

exp = ['NN', 'NNP', 'NNPS', 'NNS']
imp = ['JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
caracter = ['.', '?', '=', '>', '!', '*', '%', '<', ':', '@', ';', ',']

stp_words = stopwords.words('english')

# ler os reviews de cada documento
def read_reviews(product):
    """
    retorna um diconário com key: número do documento, value: lista de reviews
    """
    file_path_rew = './database/reviews_{}.txt'.format(product)
    
    documents_reviews = {}
    
    #lendo arquivo de reviews
    f = open(file_path_rew, 'r', encoding = "ISO-8859-1")
    rws = None
    idx = 0
    for line in tqdm(f, desc='Lendo Reviews'):
        if 'review_text: ' in line:
            rws = line[13:].split('\n')[0]
            rws = rws.split('.')
            #documents_reviews[idx] = [rw.lower() for rw in rws ]
            list_rws = []
            for rw in rws:
                if '(' in rw:
                    rw = rw.replace('(', '')
                if ')' in rw:
                    rw = rw.replace(')', '')
                if '[' in rw:
                    rw = rw.replace('[', '')
                if ']' in rw:
                    rw = rw.replace('(', '')
                if len(rw) > 0:
                    list_rws.append(rw.lower())
            
            documents_reviews[idx] = list_rws
            idx += 1
            
    f.close()
    
    return documents_reviews

def split_term(term):
    table = str.maketrans('', '', string.punctuation)
    
    aux = term.split('/')
    temp = (aux[0].translate(table).lower(), aux[1])
    return temp

def check_tag_exp(temp):
    val = False
    if (temp[0] in stp_words):
        val = False
    else:
        if len(temp[0]) > 1 and temp[1] in exp:
            val = True
    return val

def check_tag_imp(temp):
    val = False
    if (temp[0] in stp_words):
        val = False
    else:
        if len(temp[0]) > 1 and temp[1] in imp:
            val = True
    return val

def consequent_term(term, candidate, idx, idx_ver):
    idx = idx+1
    aux = term[0]
    for idy, term2 in enumerate(candidate[idx:]):
        temp2 = split_term(term2)
        if check_tag_exp(temp2):
            if term[1] == 'NN' and temp2[1] == 'NN':
                aux = aux + ' ' + temp2[0]
                idx_ver[idy] = True
                
            elif term[1] == 'NNP' and temp2[1] == 'NNP':
                aux = aux + ' ' + temp2[0]
                idx_ver[idy] = True
                
            elif term[1] == 'NN' and temp2[1] == 'NNS':
                aux = aux + ' ' + temp2[0]
                idx_ver[idy] = True
                
        else:
            break
    return aux
        
def create_dict(candidate):
    d = dict()
    for idx, v in enumerate(candidate):
        d[idx] = False
    return d

def extract_terms_tag(product):
    """
    retorna uma lista dos candidatos que serão utilizados no algoritmo
    """
    
    file_path = './database/reviews_{}.txt'.format(product)
    
    #setences_terms_tag = []
    terms_tag_exp = list()
    terms_tag_imp = list()
    
    f = open(file_path, 'r', encoding='ISO-8859-1')
    for line in tqdm(f, desc='Extraindo Candidatos'):
        if 'parsed_review: ' in line:
            candidate = line[15:].split('\n')[0]
            candidate = candidate.replace('[', '')
            candidate = candidate.replace(']', '')
            candidate = candidate.split(', ')
            
            idx_ver = create_dict(candidate)
             
            for idx, term in enumerate(candidate):
                if idx_ver[idx]:
                    continue
                else:
                    temp = split_term(term)
                    if len(temp) > 2:
                        continue
                    else:
                        if check_tag_exp(temp):
                            if idx+1 < len(candidate):
                                t = consequent_term(temp, candidate, idx, idx_ver)
                                terms_tag_exp.append(t)
                            else:       
                                terms_tag_exp.append(temp[0])
                        
                        elif check_tag_imp(temp):
                            terms_tag_imp.append(temp[0])
                    
            #setences_terms_tag.append(terms_tag)
            #print(candidate)
   
    return sorted(set(terms_tag_exp)), sorted(set(terms_tag_imp))


def documents_terms_setences(documents, candidates):
    documents_terms = {}
    
    for term in candidates:
        list_candidates = []
        for key, doc in documents.items():
            for setence in doc:
                if term in setence:
                    aux = (setence, key)
                    list_candidates.append(aux)
        if len(list_candidates) > 1:
            documents_terms[term] = list_candidates
    return documents_terms

def terms_frequencies_documents(documents, candidates):
    dict_terms_frequencies = {}
    
    for term in tqdm(candidates, desc='Calculando Frequência de Termos nos documentos'):
        cont_term = 0
        #list_doc_term = []
        for key, document in documents.items():
            for setence in document:
                if term in setence:
                    cont_term += 1
                    #list_doc_term.append(key)
                    break
        dict_terms_frequencies[term] = cont_term
    return dict_terms_frequencies

def term_frequencies_term(documents, candidates):
    matrix_terms_frequencies = {}
    
    for term1 in tqdm(candidates, desc='Calculando Matriz Frequencia de Termo a Termo'):
        matrix_terms_frequencies[term1] = {}
        for term2 in candidates:
            cont_term = 0
            if term1 != term2:
                for key, document in documents.items():
                    for setence in document:
                        if (term1 in setence) and (term2 in setence):
                            cont_term += 1
                            break
                matrix_terms_frequencies[term1][term2] = cont_term
            else:
                matrix_terms_frequencies[term1][term2] = cont_term
                
    return matrix_terms_frequencies
            
def read_aspect_cluster(product):
    file_path = './database/gold_standard/aspectCluster_{}.txt'.format(product)
    list_aspect_terms = []
    class_aspect_terms = {}
    
    f = open(file_path, 'r', encoding = "ISO-8859-1")
    for line in f:
        aspect = line.split(' ')[0]
        
        class_aspect_terms[aspect] = None
    
    keys = class_aspect_terms.keys()
    
    f = open(file_path, 'r', encoding = "ISO-8859-1")
    for line in f:
        aspect = line.split(' ')[0]
        if aspect in keys:
            class_aspect_terms[aspect] = line[len(aspect)+1:].split('\n')[0]
            

    for key, value in class_aspect_terms.items():
        terms = value[1:len(value)-1]
        terms = terms.split(', ')
        #print(terms)
        for term in terms:
            term = term.replace('_', ' ')
            list_aspect_terms.append(term)
    
    return list_aspect_terms
            

def extract_setences(documents):
    setences_without_sw = []
    
    for key, setences in documents.items():
        for setence in setences:
            text_tokens = word_tokenize(setence)
            
            # remove punctuation from each word
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in text_tokens]
            
            # remove remaining tokens that are not alphabetic
            words = [word for word in stripped if word.isalnum()]
            
            #filter out stop words
            tokens_without_sw = [word for word in words if not word in stopwords.words()]
            filtered_setence = (" ").join(tokens_without_sw)
            
            if len(filtered_setence) > 1:
                setences_without_sw.append(filtered_setence)
        
    return setences_without_sw

def extract_terms(setences):
    terms = set()
    
    for setence in setences:
        split_setence = setence.split(" ")
        for term in split_setence:
            terms.add(term)
    return list(terms)

def extract_candidates(documents_reviews):
    setences = extract_setences(documents_reviews)
    terms= extract_terms(setences)
    
    terms_explict = set()
    terms_implict = set()
    dict_term_tag = {}
    
    for term in terms:
        pos_tag = nltk.pos_tag([term])
        for tag in pos_tag:
            if tag[1] in exp:
                terms_explict.add(tag[0])
                dict_term_tag[tag[0]] = tag[1]
            elif tag[1] in imp:
                terms_implict.add(tag[0])
                dict_term_tag[tag[0]] = tag[1]
            else:
                continue
    return list(terms_explict), list(terms_implict), dict_term_tag
                
def read_json(file_path):
    with open(file_path, 'r') as f:
          data = json.load(f)
    return data

def list_save_txt(list_groups, product, name):
    with open('./resultados/{}/{}.txt'.format(product, name), 'w') as f:
        for item in list_groups:
            f.write("%s\n" % item)
            
    