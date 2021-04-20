from model.cafe import Cafe
from utils.utils import *
from preprocess.preprocess import *
from collections import OrderedDict
from graphviz import Graph
import time
import json

print("Iniciando....")

def main():
    # ler conjunto de termos candidatos
    print("Leitura Candidatos Termos")
    product = 'TV'
    documents_reviews = read_json(f'./database-process/{product}/{product}_documents_reviews.json') # ler os documentos e retorna um dicionário com um idx e uma lista com reviews

    explicit, implicit = extract_terms_tag(product)
    candidates_tag = dict()

    for k in explicit:
        candidates_tag[k] = 'NN'
    
    for k in implicit:
        candidates_tag[k] = 'VB'
    
    documents_terms = read_json(f'./database-process/{product}/{product}_doc_terms_setencas.json')
    print('-------------------------------')
    
    print("Calculando Frequências")
    frequencie_terms = read_json(f'./database-process/{product}/{product}_terms_freq_doc.json')
    
    matrix_terms = read_json(f'./database-process/{product}/{product}_terms_freq_terms.json')

    print('-------------------------------')
    
    candidates = list(frequencie_terms.keys())
    print(candidates)
        
    print("Gerando Matrizes de Associação e Similaridade")
    n = len(documents_reviews)
    mtxT = get_matrixT(candidates, frequencie_terms, matrix_terms, n)
    
    with open(f'./database-process/{product}/' + f'{product}_mtxT.json', 'w') as fp:
        json.dump(mtxT, fp)
        
    mtxG = get_matrixG(candidates)
    with open(f'./database-process/{product}/' + f'{product}_mtxG.json', 'w') as fp:
        json.dump(mtxG, fp)

    
    print('-------------------------------')
    
    print("---- Instancia modelo ---------")
    model = Cafe(candidates, candidates_tag)
    
    print('-------------------------------')
    
    print("---- Descobrindo Clusters -----")

    start = time.time()
   
    clus_aspects = model.discovery_cluster(frequencie_terms, mtxG, mtxT, documents_terms)
    print(clus_aspects)
    
    list_save_txt(clus_aspects, product, 'Clusters')
    
    print('-------------------------------')
    
    print('---- Selecionando os K clusters -------')
    results = model.select(frequencie_terms, clus_aspects)
    print(results)
    list_save_txt(results, product, f'{product}-K')
    
    end = time.time()
    print("TEMPO de Execução: ", end - start)
    #print(results)
    
main()