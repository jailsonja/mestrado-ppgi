from model.cafe import Cafe
from utils.utils import set_candidate_terms, reviews_frequencies, read_reviews_txt, get_numbers_documents, read_setences_terms, read_documents_setences, documents_terms_set
from preprocess.preprocess import get_matrixG, generate_matriz, get_matrixT, stf_word
from collections import OrderedDict
import time

print("Iniciando....")

def top100(candidates):
    freq_terms = {}
    cont = 0
    for k, c in candidates.items():
        if cont < 100:
            freq_terms[k] = c
            cont += 1
        else:
            break
        
    cnd = set(freq_terms.keys())
    return (freq_terms, cnd)

def main():
    # ler conjunto de termos candidatos
    print("Leitura Candidatos Termos")
    candidates = set_candidate_terms('Cell-phones')

    #documents_setences = read_setences_terms('Cell-phones', candidates)
    documents_setences = read_documents_setences('Cell-phones')
    documents_t = documents_terms_set(documents_setences, candidates)
    
  
    frequencie_terms, matrix_terms = reviews_frequencies('Cell-phones', candidates)
    frequencie_terms = dict(sorted(frequencie_terms.items(), key=lambda item: item[1], reverse=True))

    top = top100(frequencie_terms)
    frequencie_terms = top[0]
    candidates = top[1]
    
    
    print('-------------------------------')
    
    n = len(documents_setences)
    mtxT = get_matrixT(candidates, frequencie_terms, matrix_terms, n)
    print('-------------------------------')
    
    mtxG = get_matrixG(candidates)
    print('-------------------------------')
    
    print("---- Instancia modelo ---------")
    model = Cafe(candidates)
    print('-------------------------------')
    
    print("---- Descobrindo Clusters -----")
    start = time.time()
    
    clus_aspects = model.discovery_cluster(frequencie_terms, mtxG, mtxT, documents_t)
    print(clus_aspects)
    
    end = time.time()
    print("TEMPO de Execução: ", end - start)
    
    print('-------------------------------')
    
    print('---- Selecionando os K clusters -------')
    #results = model.select(frequencie_terms, clus_aspects)
    #print(results)
main()